package org.hiddenmoon.waterreaction.capture;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

public final class CaptureFiles {
    private CaptureFiles() {}

    public static File root(File cacheDir) throws IOException {
        File root = new File(cacheDir, "captures");
        if (!root.isDirectory() && !root.mkdirs()) {
            throw new IOException("无法创建拍照缓存目录");
        }
        return root.getCanonicalFile();
    }

    public static File create(File cacheDir) throws IOException {
        File root = root(cacheDir);
        File capture = new File(root, "capture-" + UUID.randomUUID() + ".jpg");
        if (!capture.createNewFile()) throw new IOException("无法创建拍照文件");
        return capture.getCanonicalFile();
    }

    public static boolean isUsable(File root, File file) {
        return isOwned(root, file) && file.isFile() && file.length() > 0;
    }

    public static boolean deleteOwned(File root, File file) {
        if (!isOwned(root, file)) return false;
        return !file.exists() || file.delete();
    }

    public static int deleteStale(File root, long cutoffMillis) {
        File[] children = root.listFiles();
        if (children == null) return 0;
        int deleted = 0;
        for (File child : children) {
            if (child.isFile() && child.lastModified() < cutoffMillis && deleteOwned(root, child)) {
                deleted++;
            }
        }
        return deleted;
    }

    public static boolean isOwned(File root, File file) {
        if (root == null || file == null) return false;
        try {
            File canonicalRoot = root.getCanonicalFile();
            File canonicalFile = file.getCanonicalFile();
            return canonicalRoot.equals(canonicalFile.getParentFile());
        } catch (IOException error) {
            return false;
        }
    }
}
