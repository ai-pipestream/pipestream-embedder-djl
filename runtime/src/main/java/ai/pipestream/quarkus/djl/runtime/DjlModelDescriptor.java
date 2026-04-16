package ai.pipestream.quarkus.djl.runtime;

import inference.GrpcPredictV2;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.smallrye.mutiny.Uni;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;

/**
 * Immutable snapshot of a DJL Serving model's tensor metadata, discovered
 * via the KServe v2 {@code ModelMetadata} RPC. Holds the input tensor name,
 * output tensor name, and embedding dimension so {@link DjlMutinyBatchedClient}
 * doesn't need to hardcode any of them — swap the serving name at the
 * call site and everything else auto-adjusts.
 *
 * <p>{@link #discover} returns a {@link Uni} — never blocks the caller.
 * The single RPC is bounded by {@code timeoutMs} and runs on whatever
 * executor the subscriber uses. Models whose output has a dynamic
 * ({@code -1}) hidden dimension are rejected because the client needs a
 * static dim to pre-size float arrays.
 */
public final class DjlModelDescriptor {

    private static final Logger log = LoggerFactory.getLogger(DjlModelDescriptor.class);

    private final String modelName;
    private final String inputTensorName;
    private final String outputTensorName;
    private final int dimensions;

    private DjlModelDescriptor(String modelName, String inputTensorName, String outputTensorName, int dimensions) {
        this.modelName = modelName;
        this.inputTensorName = inputTensorName;
        this.outputTensorName = outputTensorName;
        this.dimensions = dimensions;
    }

    public static Uni<DjlModelDescriptor> discover(
            MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub,
            String modelName,
            int timeoutMs) {
        GrpcPredictV2.ModelMetadataRequest req = GrpcPredictV2.ModelMetadataRequest.newBuilder()
                .setName(modelName)
                .build();

        return stub.modelMetadata(req)
                .ifNoItem().after(Duration.ofMillis(timeoutMs)).fail()
                .map(meta -> buildDescriptor(modelName, meta));
    }

    private static DjlModelDescriptor buildDescriptor(String modelName, GrpcPredictV2.ModelMetadataResponse meta) {
        if (meta.getInputsCount() == 0) {
            throw new IllegalStateException("Model " + modelName + " has no inputs");
        }
        if (meta.getOutputsCount() == 0) {
            throw new IllegalStateException("Model " + modelName + " has no outputs");
        }

        GrpcPredictV2.ModelMetadataResponse.TensorMetadata input = meta.getInputs(0);
        GrpcPredictV2.ModelMetadataResponse.TensorMetadata output = meta.getOutputs(0);

        if (!"BYTES".equals(input.getDatatype())) {
            throw new IllegalStateException("Model " + modelName + " input " + input.getName()
                    + " has datatype " + input.getDatatype() + ", expected BYTES for string input");
        }
        if (!"FP32".equals(output.getDatatype())) {
            throw new IllegalStateException("Model " + modelName + " output " + output.getName()
                    + " has datatype " + output.getDatatype() + ", expected FP32 for embeddings");
        }

        int dims = extractHiddenDim(output);
        log.info("Discovered DJL Serving model {}: input='{}' output='{}' dimensions={}",
                modelName, input.getName(), output.getName(), dims);
        return new DjlModelDescriptor(modelName, input.getName(), output.getName(), dims);
    }

    private static int extractHiddenDim(GrpcPredictV2.ModelMetadataResponse.TensorMetadata output) {
        int count = output.getShapeCount();
        if (count < 2) {
            throw new IllegalStateException("Output " + output.getName()
                    + " has shape rank " + count + ", expected at least 2 (batch, hidden_dim)");
        }
        long lastDim = output.getShape(count - 1);
        if (lastDim <= 0) {
            throw new IllegalStateException("Output " + output.getName()
                    + " has dynamic hidden dimension, cannot determine embedding size");
        }
        return (int) lastDim;
    }

    public String modelName() {
        return modelName;
    }

    public String inputTensorName() {
        return inputTensorName;
    }

    public String outputTensorName() {
        return outputTensorName;
    }

    public int dimensions() {
        return dimensions;
    }
}
