package org.hiddenmoon.waterreaction.results;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class ResultArchiveTest {
    @Rule public final TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void archiveContainsExactlyThreeMembers() throws Exception {
        byte[] jpeg = new byte[]{1, 2, 3};
        byte[] png = new byte[]{4, 5};
        byte[] json = "{}".getBytes(UTF_8);

        File archive = ResultArchive.writeAtomically(
                temporaryFolder.getRoot(), "u-1", jpeg, png, json);

        Map<String, byte[]> members = readMembers(archive);
        assertEquals(3, members.size());
        assertArrayEquals(jpeg, members.get("original.jpg"));
        assertArrayEquals(png, members.get("annotated.png"));
        assertArrayEquals(json, members.get("result.json"));
        assertFalse(new File(temporaryFolder.getRoot(), "u-1.zip.tmp").exists());
    }

    private static Map<String, byte[]> readMembers(File archive) throws Exception {
        Map<String, byte[]> members = new HashMap<>();
        try (ZipInputStream input = new ZipInputStream(new FileInputStream(archive))) {
            ZipEntry entry;
            while ((entry = input.getNextEntry()) != null) {
                members.put(entry.getName(), input.readAllBytes());
            }
        }
        return members;
    }
}
