package ai.pipestream.quarkus.djl.it;

import ai.pipestream.quarkus.djl.runtime.client.DjlServingClient;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import org.eclipse.microprofile.rest.client.RestClientBuilder;
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

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test for the DJL Serving HTTP REST path against a real
 * DJL Serving instance. Exercises:
 *
 * <ul>
 *   <li>{@link DjlServingClient#ping()} — management-API liveness</li>
 *   <li>{@link DjlServingClient#listModels()} — management-API model inventory</li>
 *   <li>{@link DjlServingClient#predict(String, JsonObject)} — inference round-trip
 *       returning a 384-dim vector for {@code all-MiniLM-L6-v2}</li>
 * </ul>
 *
 * <h2>Architecture detection</h2>
 *
 * <p>{@link #pickDjlImage()} reads {@code os.arch} at class-load time and picks
 * {@code deepjavalibrary/djl-serving:0.36.0-aarch64} on Apple Silicon / ARM64
 * Linux runners and {@code :0.36.0-cpu} on x86_64 / amd64 runners. DJL Serving
 * does not ship a multi-arch manifest — each image exists as a separate tag.
 *
 * <h2>External bypass</h2>
 *
 * <p>Pass {@code -Ddjl.host=...} (and optionally {@code -Ddjl.port=...}) on the
 * Gradle command line to skip the testcontainer and hit an already-running DJL
 * Serving instance. Useful for CI where docker-in-docker is unavailable but a
 * live DJL service is reachable.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("DJL Serving HTTP Integration Tests")
class DjlServingBackendIT {

    private static final Logger log = LoggerFactory.getLogger(DjlServingBackendIT.class);

    private static final String DJL_SERVING_VERSION = "0.36.0";

    private static final String MODEL_NAME = "all-MiniLM-L6-v2";
    private static final String HF_MODEL_URL =
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L6-v2";
    private static final int EXPECTED_DIMS = 384;
    private static final Duration TIMEOUT = Duration.ofMinutes(2);

    private static final boolean USE_EXTERNAL = System.getProperty("djl.host") != null;
    private static final String EXTERNAL_HOST = System.getProperty("djl.host", "localhost");
    private static final int EXTERNAL_PORT = Integer.parseInt(System.getProperty("djl.port", "8080"));

    private GenericContainer<?> djl;
    private DjlServingClient client;

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
        int httpPort;

        if (USE_EXTERNAL) {
            log.info("Using external DJL Serving endpoint {}:{} (container startup bypassed)",
                    EXTERNAL_HOST, EXTERNAL_PORT);
            host = EXTERNAL_HOST;
            httpPort = EXTERNAL_PORT;
        } else {
            DockerImageName image = pickDjlImage();
            log.info("Starting DJL Serving testcontainer: {}", image);
            djl = new GenericContainer<>(image)
                    .withExposedPorts(8080)
                    .waitingFor(Wait.forLogMessage(".*Inference API bind to.*\\n", 1)
                            .withStartupTimeout(Duration.ofMinutes(5)));
            djl.start();
            host = djl.getHost();
            httpPort = djl.getMappedPort(8080);

            loadModelOrFail(host, httpPort);
        }

        client = RestClientBuilder.newBuilder()
                .baseUri(URI.create("http://" + host + ":" + httpPort))
                .build(DjlServingClient.class);
    }

    @AfterAll
    void teardown() {
        if (djl != null) {
            djl.stop();
        }
    }

    @Test
    @DisplayName("/ping returns healthy response")
    void pingReturnsHealthy() {
        String pong = client.ping();
        assertThat(pong).isNotNull();
    }

    @Test
    @DisplayName("/models lists the loaded MiniLM model")
    void listModelsIncludesLoadedModel() {
        JsonObject response = client.listModels();
        JsonArray models = response.getJsonArray("models");

        assertThat(models).as("/models response should contain a 'models' array").isNotNull();
        assertThat(models).isNotEmpty();

        boolean found = false;
        for (int i = 0; i < models.size(); i++) {
            if (MODEL_NAME.equals(models.getJsonObject(i).getString("modelName"))) {
                found = true;
                break;
            }
        }
        assertThat(found).as("registry should contain '%s'", MODEL_NAME).isTrue();
    }

    @Test
    @DisplayName("/predictions/{model} returns 384-dim vectors for batch input")
    void predictReturnsVectors() {
        JsonObject body = new JsonObject().put("inputs",
                new JsonArray(List.of("hello world", "another sentence", "a third")));

        JsonArray response = client.predict(MODEL_NAME, body);

        assertThat(response).hasSize(3);
        for (int i = 0; i < response.size(); i++) {
            JsonArray vec = response.getJsonArray(i);
            assertThat(vec).hasSize(EXPECTED_DIMS);
            assertThat(vec.getNumber(0).floatValue()).isFinite();
        }
    }

    /**
     * POSTs to DJL's management API to load the model synchronously. If the load
     * fails (network egress blocked, HuggingFace rate-limited, etc.) we fail-fast
     * rather than silently skipping — the subsequent assertions would otherwise
     * produce opaque errors.
     */
    private static void loadModelOrFail(String host, int httpPort) throws Exception {
        String url = "http://" + host + ":" + httpPort
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
