package org.hiddenmoon.waterreaction.results;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class ResultArchive {
    private ResultArchive() {}

    public static File writeAtomically(
            File pendingRoot,
            String uploadId,
            byte[] originalJpeg,
            byte[] annotatedPng,
            byte[] resultJson) throws IOException {
        if (uploadId == null || !uploadId.matches("[A-Za-z0-9-]+")) {
            throw new IllegalArgumentException("uploadId 格式无效");
        }
        if (!pendingRoot.isDirectory() && !pendingRoot.mkdirs()) {
            throw new IOException("无法创建待上传目录");
        }
        File canonicalRoot = pendingRoot.getCanonicalFile();
        File temporary = new File(canonicalRoot, uploadId + ".zip.tmp");
        File target = new File(canonicalRoot, uploadId + ".zip");
        if (!canonicalRoot.equals(temporary.getCanonicalFile().getParentFile())) {
            throw new IOException("待上传路径越界");
        }
        try {
            try (FileOutputStream fileOutput = new FileOutputStream(temporary);
                 ZipOutputStream archive = new ZipOutputStream(fileOutput)) {
                write(archive, "original.jpg", originalJpeg);
                write(archive, "annotated.png", annotatedPng);
                write(archive, "result.json", resultJson);
                archive.finish();
                fileOutput.getFD().sync();
            }
            try {
                Files.move(temporary.toPath(), target.toPath(),
                        StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException error) {
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
            }
            return target;
        } catch (IOException | RuntimeException error) {
            Files.deleteIfExists(temporary.toPath());
            throw error;
        }
    }

    private static void write(ZipOutputStream archive, String name, byte[] content) throws IOException {
        if (content == null || content.length == 0) throw new IOException(name + " 内容为空");
        archive.putNextEntry(new ZipEntry(name));
        archive.write(content);
        archive.closeEntry();
    }
}
