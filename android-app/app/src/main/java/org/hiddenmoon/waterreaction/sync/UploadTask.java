package org.hiddenmoon.waterreaction.sync;

import java.io.File;

public final class UploadTask {
    public final String uploadId;
    public final File archive;
    public final String status;
    public final int attempts;
    public final long nextAttemptAt;
    public final String lastError;

    UploadTask(
            String uploadId,
            File archive,
            String status,
            int attempts,
            long nextAttemptAt,
            String lastError) {
        this.uploadId = uploadId;
        this.archive = archive;
        this.status = status;
        this.attempts = attempts;
        this.nextAttemptAt = nextAttemptAt;
        this.lastError = lastError;
    }
}
