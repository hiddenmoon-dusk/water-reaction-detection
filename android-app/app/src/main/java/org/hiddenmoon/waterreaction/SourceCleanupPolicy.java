package org.hiddenmoon.waterreaction;

public final class SourceCleanupPolicy {
    private SourceCleanupPolicy() {}

    public static boolean shouldDeleteCapture(
            String photoSource, boolean archiveCreated) {
        return archiveCreated && "camera".equals(photoSource);
    }
}
