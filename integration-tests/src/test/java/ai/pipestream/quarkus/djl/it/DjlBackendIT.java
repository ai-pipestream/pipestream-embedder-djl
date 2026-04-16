package ai.pipestream.quarkus.djl.it;

import ai.pipestream.quarkus.djl.runtime.DjlBackend;
import ai.pipestream.quarkus.djl.runtime.DjlModelDescriptor;
import ai.pipestream.quarkus.djl.runtime.DjlMutinyBatchedClient;
import inference.MutinyGRPCInferenceServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.DockerImageName;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for {@link DjlBackend} against a real DJL Serving
 * instance speaking KServe v2 gRPC. Exercises:
 *
 * <ul>
 *   <li>{@link DjlBackend#supports(String)} — returns {@code Uni<Boolean>}
 *       resolving to {@code true} for a loaded model, {@code false} for an
 *       unknown one. No silent swallow, no optimistic lie.</li>
 *   <li>{@link DjlBackend#embed(String, List)} — round-trips actual text
 *       through the DJL container and returns model-shaped float[] vectors.</li>
 *   <li>{@link DjlModelDescriptor#discover} — metadata RPC returns tensor
 *       names and dim count.</li>
 * </ul>
 *
 * <h2>Architecture detection</h2>
 *
 * <p>{@link #pickDjlImage()} reads {@code os.arch} at class-load time and
 * picks {@code deepjavalibrary/djl-serving:0.36.0-aarch64} on Apple Silicon
 * / ARM64 Linux runners and {@code :0.36.0-cpu} on x86_64 / amd64 runners.
 * DJL Serving does not ship a multi-arch manifest; each image exists as a
 * separate tag on Docker Hub.
 *
 * <h2>External bypass</h2>
 *
 * <p>Pass {@code -Ddjl.host=...} (and optionally {@code -Ddjl.port=...})
 * on the Gradle command line to skip the testcontainer and hit an already-
 * running DJL Serving instance. Useful for CI where docker-in-docker is
 * unavailable but a live DJL service is reachable.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("DJL Backend Integration Tests")
class DjlBackendIT {

    private static final Logger log = LoggerFactory.getLogger(DjlBackendIT.class);

    /** DJL Serving version pinned here for dependabot's docker ecosystem to bump. */
    private static final String DJL_SERVING_VERSION = "0.36.0";

    /** Model loaded for the tests — small (22M params), fast on CPU, well-known. */
    private static final String MODEL_NAME = "all-MiniLM-L6-v2";
    private static final String HF_MODEL_URL =
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2";
    private static final int EXPECTED_DIMS = 384;
    private static final int TIMEOUT_MS = 120_000;

    private static final boolean USE_EXTERNAL = System.getProperty("djl.host") != null;
    private static final String EXTERNAL_HOST = System.getProperty("djl.host", "localhost");
    private static final int EXTERNAL_PORT = Integer.parseInt(System.getProperty("djl.port", "8080"));

    private GenericContainer<?> djl;
    private ManagedChannel channel;
    private MutinyGRPCInferenceServiceGrpc.MutinyGRPCInferenceServiceStub stub;

    /**
     * Pick the right DJL Serving image tag for the test runner's architecture.
     * DJL Serving publishes separate tags per arch (no multi-arch manifest).
     */
    static DockerImageName pickDjlImage() {
        String arch = System.getProperty("os.arch", "").toLowerCase();
        String tag = (arch.contains("aarch64") || arch.contains("arm64"))
                ? DJL_SERVING_VERSION + "-aarch64"
                : DJL_SERVING_VERSION + "-cpu";
        return DockerImageName.parse("deepjavalibrary/djl-serving:" + tag);
    }

    @BeforeAll
    void setup() throws Exception {
        String host;
        int grpcPort;

        if (USE_EXTERNAL) {
            log.info("Using external DJL Serving endpoint {}:{} (container startup bypassed)",
                    EXTERNAL_HOST, EXTERNAL_PORT);
            host = EXTERNAL_HOST;
            grpcPort = EXTERNAL_PORT;
        } else {
            DockerImageName image = pickDjlImage();
            log.info("Starting DJL Serving testcontainer: {}", image);
            djl = new GenericContainer<>(image)
                    .withExposedPorts(8080, 8082)  // 8080 management, 8082 KServe v2 gRPC
                    .withEnv("DJL_SERVING_GRPC_PORT", "8082")
                    .waitingFor(Wait.forLogMessage(".*Inference API bind to.*\\n", 1)
                            .withStartupTimeout(Duration.ofMinutes(5)));
            djl.start();
            host = djl.getHost();
            grpcPort = djl.getMappedPort(8082);

            // Load the model through DJL's management REST API on 8080.
            int managementPort = djl.getMappedPort(8080);
            loadModelOrSkip(host, managementPort);
        }

        channel = ManagedChannelBuilder.forAddress(host, grpcPort)
                .usePlaintext()
                .build();
        stub = MutinyGRPCInferenceServiceGrpc.newMutinyStub(channel);
    }

    @AfterAll
    void teardown() throws InterruptedException {
        if (channel != null) {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
        if (djl != null) {
            djl.stop();
        }
    }

    @Test
    @DisplayName("ModelMetadata discovery returns expected input/output/dims")
    void discoverReturnsDescriptor() {
        DjlModelDescriptor descriptor = DjlModelDescriptor.discover(stub, MODEL_NAME, TIMEOUT_MS)
                .await().atMost(Duration.ofMillis(TIMEOUT_MS));

        assertThat(descriptor.modelName()).isEqualTo(MODEL_NAME);
        assertThat(descriptor.dimensions()).isEqualTo(EXPECTED_DIMS);
        assertThat(descriptor.inputTensorName()).isNotBlank();
        assertThat(descriptor.outputTensorName()).isNotBlank();
    }

    @Test
    @DisplayName("embed() returns one float[384] per input text, in order")
    void embedReturnsVectors() {
        DjlMutinyBatchedClient client = DjlMutinyBatchedClient.create(stub, MODEL_NAME, 8, TIMEOUT_MS)
                .await().atMost(Duration.ofMillis(TIMEOUT_MS));

        List<float[]> vectors = client.embed(List.of("hello world", "another sentence", "a third"))
                .await().atMost(Duration.ofMillis(TIMEOUT_MS));

        assertThat(vectors).hasSize(3);
        for (float[] v : vectors) {
            assertThat(v).hasSize(EXPECTED_DIMS);
            assertThat(Float.isFinite(v[0])).isTrue();
        }
    }

    /**
     * Verify {@link DjlBackend#supports(String)} is honest:
     * - returns Uni that resolves true for a loaded model
     * - returns Uni that resolves false for an unknown model (real probe,
     *   not an optimistic lie)
     *
     * <p>Can't reuse the @Inject stub because this IT isn't @QuarkusTest
     * (DJL testcontainer startup is incompatible with the Quarkus dev-services
     * lifecycle). Instead construct DjlBackend by hand and inject the test
     * channel's stub via reflection — the backend doesn't know the difference.
     */
    @Test
    @DisplayName("supports(loaded model) resolves to true reactively")
    void supportsLoadedModelReturnsTrue() throws Exception {
        DjlBackend backend = newBackendWithTestStub();

        Boolean supported = backend.supports(MODEL_NAME).await().atMost(Duration.ofMillis(TIMEOUT_MS));

        assertThat(supported)
                .as("loaded model must probe true via real ModelMetadata RPC")
                .isTrue();
    }

    @Test
    @DisplayName("supports(unknown model) resolves to false reactively — no silent lie")
    void supportsUnknownModelReturnsFalse() throws Exception {
        DjlBackend backend = newBackendWithTestStub();

        Boolean supported = backend.supports("nonexistent-model-xyz")
                .await().atMost(Duration.ofMillis(TIMEOUT_MS));

        assertThat(supported)
                .as("unknown model must probe false — the old fire-and-forget boolean API "
                        + "would have returned true here and fooled the router")
                .isFalse();
    }

    // -------- helpers --------

    private DjlBackend newBackendWithTestStub() throws Exception {
        DjlBackend backend = new DjlBackend();
        var stubField = DjlBackend.class.getDeclaredField("stub");
        stubField.setAccessible(true);
        stubField.set(backend, stub);
        var batchSizeField = DjlBackend.class.getDeclaredField("batchSize");
        batchSizeField.setAccessible(true);
        batchSizeField.setInt(backend, 8);
        var timeoutField = DjlBackend.class.getDeclaredField("timeoutMs");
        timeoutField.setAccessible(true);
        timeoutField.setInt(backend, TIMEOUT_MS);
        return backend;
    }

    /**
     * POSTs to DJL's management API to load the model. If the load fails
     * (e.g. network egress blocked, HuggingFace rate-limited), we mark
     * the test container unusable and the subsequent assertions will
     * surface the actual DJL error — better than a silent skip.
     */
    private static void loadModelOrSkip(String host, int managementPort) throws Exception {
        String url = "http://" + host + ":" + managementPort
                + "/models?url=" + java.net.URLEncoder.encode(HF_MODEL_URL, "UTF-8")
                + "&model_name=" + MODEL_NAME
                + "&engine=PyTorch&synchronous=true";
        log.info("Loading model via {}", url);

        HttpClient http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofMinutes(5))
                .POST(HttpRequest.BodyPublishers.noBody())
                .build();

        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 300) {
            throw new IllegalStateException("DJL model load failed: HTTP "
                    + resp.statusCode() + " — " + resp.body());
        }
        log.info("Model loaded: {}", resp.body());
    }
}
