package ai.pipestream.quarkus.djl.runtime;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.HealthCheckResponseBuilder;
import org.eclipse.microprofile.health.Readiness;

import java.util.Set;

/**
 * MicroProfile readiness check for DJL Serving availability. Reports
 * DOWN until {@link DjlModelRegistry} has seen at least one model
 * transition to {@code READY} with a successful embedding probe.
 * Liveness is unaffected — the app stays alive regardless of DJL
 * Serving state.
 */
@Readiness
@ApplicationScoped
public class DjlServingReadinessCheck implements HealthCheck {

    @Inject
    DjlModelRegistry modelRegistry;

    @Override
    public HealthCheckResponse call() {
        boolean reachable = modelRegistry.isDjlServingReachable();
        Set<String> readyModels = modelRegistry.getReadyModelNames();
        boolean ready = reachable && !readyModels.isEmpty();

        HealthCheckResponseBuilder builder = HealthCheckResponse.named("DJL Serving")
                .status(ready)
                .withData("reachable", reachable)
                .withData("readyModels", String.join(", ", readyModels))
                .withData("lastRefresh", modelRegistry.getLastRefresh().toString());

        if (!reachable) {
            builder.withData("message", "DJL Serving is not reachable — waiting for connection");
        } else if (readyModels.isEmpty()) {
            builder.withData("message", "DJL Serving is reachable but no models are READY");
        }

        return builder.build();
    }
}
