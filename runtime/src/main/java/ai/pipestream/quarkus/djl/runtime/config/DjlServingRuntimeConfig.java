package ai.pipestream.quarkus.djl.runtime.config;

import io.quarkus.runtime.annotations.ConfigPhase;
import io.quarkus.runtime.annotations.ConfigRoot;
import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;

import java.time.Duration;

/**
 * Runtime configuration for the DJL Serving embedding extension.
 *
 * <p>Bound under {@code pipestream.djl-serving.*} in application.properties.
 * The REST client URL is configured separately via
 * {@code quarkus.rest-client.djl-serving.url}; both should point at the
 * same DJL Serving base URL.
 */
@ConfigMapping(prefix = "pipestream.djl-serving")
@ConfigRoot(phase = ConfigPhase.RUN_TIME)
public interface DjlServingRuntimeConfig {

    /**
     * Whether the DJL Serving backend is enabled. When {@code false} the
     * scheduled registry refresh is skipped and the backend reports no
     * ready models.
     */
    @WithDefault("true")
    boolean enabled();

    /**
     * Per-request timeout for management / prediction calls used by the
     * scheduled refresh. Does not affect the {@code @RestClient} direct
     * inference path, which uses its own timeout configuration under
     * {@code quarkus.rest-client.djl-serving.*}.
     */
    @WithDefault("10s")
    Duration requestTimeout();

    /**
     * Refresh interval for polling DJL Serving's {@code /models} endpoint
     * and re-probing unverified models. Consumed via property expression
     * in {@code @Scheduled(every = "{pipestream.djl-serving.refresh-interval}")}.
     */
    @WithDefault("30s")
    Duration refreshInterval();
}
