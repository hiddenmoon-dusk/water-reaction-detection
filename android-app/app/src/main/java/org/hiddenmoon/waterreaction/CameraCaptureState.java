package org.hiddenmoon.waterreaction;

final class CameraCaptureState {
    private String pendingUri;

    void begin(String uri) {
        pendingUri = uri;
    }

    void restore(String uri) {
        pendingUri = uri;
    }

    String pendingUri() {
        return pendingUri;
    }

    String consume(boolean resultOk, boolean imageWasWritten) {
        String capturedUri = pendingUri;
        pendingUri = null;
        return resultOk || imageWasWritten ? capturedUri : null;
    }
}
