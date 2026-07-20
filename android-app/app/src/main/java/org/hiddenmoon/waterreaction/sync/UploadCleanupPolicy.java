package org.hiddenmoon.waterreaction.sync;

public final class UploadCleanupPolicy {
    private UploadCleanupPolicy() {}

    public static boolean mayDeleteArchive(UploadDecision decision) {
        return decision == UploadDecision.SUCCESS;
    }

    public static boolean mayDeleteQueueRow(
            UploadDecision decision, boolean archiveDeleted) {
        return decision == UploadDecision.SUCCESS && archiveDeleted;
    }
}
