package ai.pipestream.quarkus.djl.runtime;

import ai.pipestream.quarkus.djl.runtime.client.DjlServingClient;
import ai.pipestream.quarkus.djl.runtime.config.DjlServingRuntimeConfig;
import io.quarkus.scheduler.Scheduled;
import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.eclipse.microprofile.rest.client.inject.RestClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Health-checking registry for models hosted on DJL Serving.
 *
 * <p>Polls the configured DJL Serving URL on a schedule to discover
 * loaded models. For each discovered model, runs a real embedding
 * probe (sample text → verify a vector is returned) so readiness
 * reflects actual inference health, not just management-API presence.
 *
 * <p>The refresh interval is governed by
 * {@code pipestream.djl-serving.refresh-interval}. On startup the
 * registry eagerly refreshes once so consumers don't race the 5-second
 * scheduler delay.
 */
@ApplicationScoped
public class DjlModelRegistry {

    private static final Logger log = LoggerFactory.getLogger(DjlModelRegistry.class);

    private static final String PROBE_TEXT = "health check embedding probe";

    private final Instance<DjlServingClient> clientInstance;
    private final DjlServingRuntimeConfig config;

    private final Map<String, ModelStatus> models = new ConcurrentHashMap<>();

    private volatile Instant lastRefresh = Instant.EPOCH;
    private volatile boolean djlServingReachable = false;

    /**
     * Model status captured from DJL Serving's {@code /models} response,
     * enriched with the result of a real embedding probe.
     *
     * @param name         DJL Serving model name
     * @param status       management-API status ({@code READY}, {@code LOADING}, etc.)
     * @param modelUrl     registration URL reported by DJL Serving
     * @param lastSeen     last registry refresh that observed this model
     * @param predictOk    embedding probe result ({@code null} = not yet probed)
     * @param predictError last probe failure message, or {@code null}
     * @param probeDims    vector dimensionality returned by the probe (0 if unknown)
     */
    public record ModelStatus(String name, String status, String modelUrl, Instant lastSeen,
                              Boolean predictOk, String predictError, int probeDims) {

        public boolean isRegistered() {
            return "READY".equalsIgnoreCase(status) || "Healthy".equalsIgnoreCase(status);
        }

        public boolean isReady() {
            return isRegistered() && Boolean.TRUE.equals(predictOk);
        }

        public boolean isError() {
            return isRegistered() && Boolean.FALSE.equals(predictOk);
        }
    }

    @Inject
    public DjlModelRegistry(@RestClient Instance<DjlServingClient> clientInstance,
                            DjlServingRuntimeConfig config) {
        this.clientInstance = clientInstance;
        this.config = config;
    }

    void onStart(@Observes io.quarkus.runtime.StartupEvent ev) {
        if (!config.enabled()) {
            log.info("DJL Serving extension disabled; skipping eager refresh");
            return;
        }
        log.info("Eager DJL model registry refresh on startup");
        refresh();
    }

    @Scheduled(every = "{pipestream.djl-serving.refresh-interval:30s}", delayed = "5s",
            concurrentExecution = Scheduled.ConcurrentExecution.SKIP)
    void scheduledRefresh() {
        if (!config.enabled()) {
            return;
        }
        refresh();
    }

    public void refresh() {
        if (clientInstance.isUnsatisfied()) {
            log.debug("DJL Serving client not available, skipping refresh");
            return;
        }

        try {
            DjlServingClient client = clientInstance.get();
            JsonObject response = client.listModels().await().atMost(config.requestTimeout());

            JsonArray modelArray = response.getJsonArray("models");
            if (modelArray == null || modelArray.isEmpty()) {
                log.warn("DJL Serving /models returned no models");
                djlServingReachable = true;
                return;
            }

            Set<String> seen = ConcurrentHashMap.newKeySet();

            for (int i = 0; i < modelArray.size(); i++) {
                JsonObject m = modelArray.getJsonObject(i);
                String name = m.getString("modelName");
                String status = m.getString("status", "UNKNOWN");
                String modelUrl = m.getString("modelUrl", "");

                seen.add(name);
                ModelStatus prev = models.get(name);

                if ("READY".equalsIgnoreCase(status) || "Healthy".equalsIgnoreCase(status)) {
                    boolean needsProbe = prev == null
                            || prev.predictOk() == null
                            || Boolean.FALSE.equals(prev.predictOk());

                    ModelStatus ms = needsProbe
                            ? probeEmbedding(client, name, status, modelUrl)
                            : new ModelStatus(name, status, modelUrl, Instant.now(),
                                    prev.predictOk(), prev.predictError(), prev.probeDims());

                    models.put(name, ms);

                    if (prev == null) {
                        log.info("Model '{}' discovered: status={}, predictOk={}, dims={}",
                                name, ms.status(), ms.predictOk(), ms.probeDims());
                    } else if (!prev.status().equals(ms.status())
                            || !Objects.equals(prev.predictOk(), ms.predictOk())) {
                        log.info("Model '{}' status changed: {}(ok={}) -> {}(ok={}, dims={})",
                                name, prev.status(), prev.predictOk(),
                                ms.status(), ms.predictOk(), ms.probeDims());
                    }
                } else {
                    models.put(name, new ModelStatus(name, status, modelUrl, Instant.now(),
                            null, null, 0));
                    if (prev == null) {
                        log.info("Model '{}' discovered: status={} (not ready)", name, status);
                    }
                }
            }

            models.keySet().removeIf(name -> {
                if (!seen.contains(name)) {
                    log.info("Model '{}' no longer on DJL Serving — removing from registry", name);
                    return true;
                }
                return false;
            });

            djlServingReachable = true;
            lastRefresh = Instant.now();
            log.info("DJL model registry refreshed: {} models total, {} ready",
                    models.size(), getReadyModelNames().size());

        } catch (Exception e) {
            djlServingReachable = false;
            log.warn("Failed to refresh DJL model registry: {}", e.getMessage());
        }
    }

    private ModelStatus probeEmbedding(DjlServingClient client, String name,
                                        String status, String modelUrl) {
        try {
            JsonObject input = new JsonObject().put("inputs", PROBE_TEXT);
            JsonArray result = client.predict(name, input).await().atMost(config.requestTimeout());

            if (result == null || result.isEmpty()) {
                return new ModelStatus(name, status, modelUrl, Instant.now(),
                        false, "Embedding probe returned empty result", 0);
            }

            int dims;
            if (result.getValue(0) instanceof JsonArray inner) {
                dims = inner.size();
            } else {
                dims = result.size();
            }

            if (dims == 0) {
                return new ModelStatus(name, status, modelUrl, Instant.now(),
                        false, "Embedding probe returned zero-dimension vector", 0);
            }

            log.info("Model '{}' embedding probe succeeded: {}d vector", name, dims);
            return new ModelStatus(name, status, modelUrl, Instant.now(), true, null, dims);

        } catch (Exception e) {
            String err = e.getCause() != null ? e.getCause().getMessage() : e.getMessage();
            log.warn("Model '{}' embedding probe failed: {}", name, err);
            return new ModelStatus(name, status, modelUrl, Instant.now(), false, err, 0);
        }
    }

    public Set<String> getReadyModelNames() {
        Set<String> ready = ConcurrentHashMap.newKeySet();
        for (ModelStatus ms : models.values()) {
            if (ms.isReady()) {
                ready.add(ms.name());
            }
        }
        return Collections.unmodifiableSet(ready);
    }

    public Map<String, ModelStatus> getAllModels() {
        return Collections.unmodifiableMap(models);
    }

    public boolean isModelReady(String modelName) {
        ModelStatus ms = models.get(modelName);
        return ms != null && ms.isReady();
    }

    public boolean isDjlServingReachable() {
        return djlServingReachable;
    }

    public Instant getLastRefresh() {
        return lastRefresh;
    }
}
