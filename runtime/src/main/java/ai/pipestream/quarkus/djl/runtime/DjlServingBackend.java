package ai.pipestream.quarkus.djl.runtime;

import ai.pipestream.module.embedder.spi.EmbeddingBackend;
import ai.pipestream.quarkus.djl.runtime.client.DjlServingClient;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.eclipse.microprofile.rest.client.inject.RestClient;

import java.util.ArrayList;
import java.util.List;

/**
 * DJL Serving backend for the {@link EmbeddingBackend} SPI, implemented
 * over DJL Serving's HTTP REST API via MicroProfile REST Client.
 *
 * <p>Uses {@link Singleton} (not {@code @ApplicationScoped}) to avoid ARC
 * client-proxy generation — the SPI interface lives in a separate
 * {@code module-embedder-api} jar, and client proxies can't cast across
 * the extension classloader boundary when discovered via
 * {@code Instance<EmbeddingBackend>}.
 *
 * <p><b>Concurrency model.</b> Plain blocking REST calls. The SPI is
 * synchronous; callers (e.g. {@code EmbedderGrpcImpl}) are expected to
 * invoke from a {@code @RunOnVirtualThread} entry point so the carrier
 * is parked on the HTTP wait rather than burning a platform thread.
 *
 * <p>Model name resolution (HuggingFace identifier → short name → DJL
 * serving name) is the caller's responsibility. {@link #supports(String)}
 * just asks {@link DjlModelRegistry} whether a model with exactly that
 * name is loaded and has passed an embedding probe.
 *
 * <p>Configuration:
 * <pre>
 * quarkus.rest-client.djl-serving.url=http://localhost:8080
 * pipestream.djl-serving.refresh-interval=30s
 * </pre>
 */
@Singleton
public class DjlServingBackend implements EmbeddingBackend {

    @Inject
    @RestClient
    DjlServingClient client;

    @Inject
    DjlModelRegistry modelRegistry;

    @Override
    public String name() {
        return "djl-serving";
    }

    @Override
    public boolean supports(String servingName) {
        if (servingName == null || servingName.isBlank()) {
            return false;
        }
        return modelRegistry.isModelReady(servingName);
    }

    @Override
    public List<float[]> embed(String servingName, List<String> texts) {
        if (texts == null || texts.isEmpty()) {
            return List.of();
        }
        JsonObject body = new JsonObject().put("inputs", new JsonArray(texts));
        // Quarkus REST Client throws WebApplicationException for HTTP errors (4xx/5xx)
        // and ProcessingException for connection-level failures (refused, timeout, TLS).
        // Both are RuntimeException subclasses; let them propagate so the router /
        // retry policy can classify them and decide failover vs retry.
        JsonArray response = client.predict(servingName, body);
        return parseBatch(response);
    }

    /**
     * Parses DJL Serving's batch response. Handles both nested
     * ({@code [[f1,f2,...],[f1,f2,...]]}) and flat ({@code [f1,f2,...]})
     * shapes — some DJL model handlers return a single vector flat when
     * given a single input, nested otherwise.
     */
    private static List<float[]> parseBatch(JsonArray response) {
        if (response == null || response.isEmpty()) {
            return List.of();
        }
        Object first = response.getValue(0);
        if (first instanceof JsonArray) {
            List<float[]> out = new ArrayList<>(response.size());
            for (int i = 0; i < response.size(); i++) {
                JsonArray vec = response.getJsonArray(i);
                float[] arr = new float[vec.size()];
                for (int j = 0; j < vec.size(); j++) {
                    arr[j] = vec.getNumber(j).floatValue();
                }
                out.add(arr);
            }
            return out;
        }
        // Flat response — single-vector batch
        float[] arr = new float[response.size()];
        for (int i = 0; i < response.size(); i++) {
            arr[i] = response.getNumber(i).floatValue();
        }
        return List.of(arr);
    }
}
