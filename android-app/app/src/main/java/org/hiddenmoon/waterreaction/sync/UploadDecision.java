package org.hiddenmoon.waterreaction.sync;

public enum UploadDecision {
    SUCCESS,
    RETRY,
    BLOCKED;

    public static UploadDecision forHttp(int statusCode) {
        if (statusCode == 201 || statusCode == 208) return SUCCESS;
        if (statusCode == 408 || statusCode == 425 || statusCode == 429
                || statusCode >= 500 && statusCode <= 599) return RETRY;
        return BLOCKED;
    }
}
