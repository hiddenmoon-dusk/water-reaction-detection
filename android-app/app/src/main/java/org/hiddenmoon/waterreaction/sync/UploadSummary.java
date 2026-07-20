package org.hiddenmoon.waterreaction.sync;

/** A content-free snapshot of the durable result upload queue. */
public final class UploadSummary {
    public final int pending;
    public final int uploading;
    public final int retryWait;
    public final int blocked;
    public final String latestError;

    public UploadSummary(
            int pending,
            int uploading,
            int retryWait,
            int blocked,
            String latestError) {
        this.pending = Math.max(0, pending);
        this.uploading = Math.max(0, uploading);
        this.retryWait = Math.max(0, retryWait);
        this.blocked = Math.max(0, blocked);
        this.latestError = latestError;
    }

    public int total() {
        return pending + uploading + retryWait + blocked;
    }

    public boolean canRetry() {
        return pending > 0 || retryWait > 0;
    }
}
