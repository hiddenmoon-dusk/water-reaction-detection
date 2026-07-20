package org.hiddenmoon.waterreaction.sync;

public final class UploadCompatibilityPolicy {
    public static final String INVALID_PAYLOAD = "invalid_payload";

    private UploadCompatibilityPolicy() {}

    public static boolean shouldRequeue(String errorCode) {
        return INVALID_PAYLOAD.equals(errorCode);
    }
}
