package org.hiddenmoon.waterreaction.capture;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;

public class CaptureFilesTest {
    @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void deletesOnlyFilesInsideCaptureRoot() throws Exception {
        File root = temporaryFolder.newFolder("captures");
        File captured = new File(root, "capture.jpg");
        assertTrue(captured.createNewFile());
        assertTrue(CaptureFiles.deleteOwned(root, captured));
        assertFalse(captured.exists());

        File gallery = temporaryFolder.newFile("gallery.jpg");
        assertFalse(CaptureFiles.deleteOwned(root, gallery));
        assertTrue(gallery.exists());
    }

    @Test
    public void usableCaptureMustContainBytes() throws Exception {
        File root = temporaryFolder.newFolder("captures");
        File empty = new File(root, "empty.jpg");
        assertTrue(empty.createNewFile());
        assertFalse(CaptureFiles.isUsable(root, empty));

        File written = new File(root, "written.jpg");
        try (FileOutputStream output = new FileOutputStream(written)) {
            output.write(new byte[]{1, 2, 3});
        }
        assertTrue(CaptureFiles.isUsable(root, written));
    }

    @Test
    public void staleCleanupNeverLeavesCaptureRoot() throws Exception {
        File root = temporaryFolder.newFolder("captures");
        File stale = new File(root, "stale.jpg");
        assertTrue(stale.createNewFile());
        assertTrue(stale.setLastModified(1));
        File gallery = temporaryFolder.newFile("gallery.jpg");
        assertTrue(gallery.setLastModified(1));

        assertTrue(CaptureFiles.deleteStale(root, 2) >= 1);
        assertFalse(stale.exists());
        assertTrue(gallery.exists());
    }
}
