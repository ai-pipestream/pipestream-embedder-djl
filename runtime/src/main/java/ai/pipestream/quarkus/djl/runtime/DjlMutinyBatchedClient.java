package ai.pipestream.quarkus.djl.runtime;

import com.google.protobuf.ByteString;
import inference.GrpcPredictV2;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Fully-reactive batched client for DJL Serving's KServe v2 gRPC inference.
 * Mutiny all the way down: no {@code Uni.createFrom().item(() -> blockingCall)}
 * wrappers, no blocking stubs. Every RPC returns {@code Uni<T>}; a run of
 * N texts is sliced into {@code ceil(N/batchSize)} sub-batches that fire
 * concurrently via {@link Multi#transformToUniAndMerge}, each writing into
 * disjoint slots of a shared output array under Mutiny's happens-before
 * boundary at {@code collect().asList()}.
 */
public final class DjlMutinyBatchedClient {

    private static final Logger log = LoggerFactory.getLogger(DjlMutinyBatchedClient.class);

    private final MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;
    private final DjlModelDescriptor descriptor;
    private final int batchSize;
    private final Duration requestTimeout;

    /**
     * Asynchronously build a client — kicks off a {@code ModelMetadata} RPC
     * to discover tensor names + dims, then constructs the client. Never
     * blocks the caller.
     */
    public static Uni<DjlMutinyBatchedClient> create(
            MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
            String modelName, int batchSize, int timeoutMs) {
        return DjlModelDescriptor.discover(stub, modelName, timeoutMs)
                .map(descriptor -> new DjlMutinyBatchedClient(stub, descriptor, batchSize, timeoutMs));
    }

    public DjlMutinyBatchedClient(MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
                                  DjlModelDescriptor descriptor,
                                  int batchSize, int timeoutMs) {
        this.stub = stub.withCompression("gzip");
        this.descriptor = descriptor;
        this.batchSize = batchSize;
        this.requestTimeout = Duration.ofMillis(timeoutMs);

        log.info("Created DJL Mutiny Batched client for model: {} (batch_size={}, dims={}, timeout={}ms)",
                descriptor.modelName(), batchSize, descriptor.dimensions(), timeoutMs);
    }

    public Uni<List<float[]>> embed(List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return Uni.createFrom().item(List.of());
        }
        final int total = texts.size();
        final int numBatches = (total + batchSize - 1) / batchSize;
        final float[][] out = new float[total][];

        return Multi.createFrom().range(0, numBatches)
                .onItem().transformToUniAndMerge(batchIdx -> {
                    final int from = batchIdx * batchSize;
                    final int to = Math.min(from + batchSize, total);
                    final List<String> batchTexts = new ArrayList<>(texts.subList(from, to));

                    GrpcPredictV2.ModelInferRequest request = buildRequest(batchTexts);
                    return stub.modelInfer(request)
                            .ifNoItem().after(requestTimeout).fail()
                            .map(response -> {
                                List<float[]> slice = new ArrayList<>(batchTexts.size());
                                extractEmbeddings(response, batchTexts.size(), slice);
                                for (int i = 0; i < slice.size(); i++) {
                                    out[from + i] = slice.get(i);
                                }
                                return batchIdx;
                            });
                })
                .collect().asList()
                .map(ignored -> Arrays.asList(out));
    }

    private GrpcPredictV2.ModelInferRequest buildRequest(List<String> batchTexts) {
        GrpcPredictV2.InferTensorContents.Builder contents = GrpcPredictV2.InferTensorContents.newBuilder();
        for (String text : batchTexts) {
            contents.addBytesContents(ByteString.copyFromUtf8(text));
        }
        GrpcPredictV2.ModelInferRequest.InferInputTensor inputTensor =
                GrpcPredictV2.ModelInferRequest.InferInputTensor.newBuilder()
                        .setName(descriptor.inputTensorName())
                        .setDatatype("BYTES")
                        .addShape(batchTexts.size())
                        .setContents(contents)
                        .build();
        GrpcPredictV2.ModelInferRequest.InferRequestedOutputTensor outputTensor =
                GrpcPredictV2.ModelInferRequest.InferRequestedOutputTensor.newBuilder()
                        .setName(descriptor.outputTensorName())
                        .build();
        return GrpcPredictV2.ModelInferRequest.newBuilder()
                .setModelName(descriptor.modelName())
                .setModelVersion("")
                .addInputs(inputTensor)
                .addOutputs(outputTensor)
                .build();
    }

    private void extractEmbeddings(GrpcPredictV2.ModelInferResponse response, int batchSize, List<float[]> out) {
        int dims = descriptor.dimensions();

        if (response.getRawOutputContentsCount() > 0) {
            ByteString raw = response.getRawOutputContents(0);
            FloatBuffer fb = raw.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            int expected = batchSize * dims;
            if (fb.remaining() < expected) {
                throw new IllegalStateException("Response raw_output_contents too small: got "
                        + fb.remaining() + " floats, expected " + expected);
            }
            for (int i = 0; i < batchSize; i++) {
                float[] embedding = new float[dims];
                fb.get(embedding);
                out.add(embedding);
            }
            return;
        }

        if (response.getOutputsCount() > 0) {
            GrpcPredictV2.ModelInferResponse.InferOutputTensor tensor = response.getOutputs(0);
            if (tensor.hasContents()) {
                List<Float> fp = tensor.getContents().getFp32ContentsList();
                for (int i = 0; i < batchSize; i++) {
                    float[] embedding = new float[dims];
                    for (int j = 0; j < dims; j++) {
                        embedding[j] = fp.get(i * dims + j);
                    }
                    out.add(embedding);
                }
                return;
            }
        }

        throw new IllegalStateException("ModelInfer response had neither raw_output_contents nor contents.fp32_contents");
    }

    public DjlModelDescriptor getDescriptor() {
        return descriptor;
    }
}
