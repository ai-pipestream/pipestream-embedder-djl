# DJL Serving Integration Tests

Exercise `DjlBackend` and its underlying Mutiny batched client against a real
DJL Serving instance via KServe v2 gRPC.

By default a testcontainer launches `deepjavalibrary/djl-serving` locally.
Point at an external DJL Serving endpoint instead with:

```bash
./gradlew :integration-tests:test \
    -Ddjl.host=djl.internal \
    -Ddjl.port=8082
```
