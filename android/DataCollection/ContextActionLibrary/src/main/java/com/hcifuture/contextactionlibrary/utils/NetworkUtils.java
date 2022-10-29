package com.hcifuture.contextactionlibrary.utils;

import android.content.Context;
import android.net.Uri;
import android.util.Log;
import android.webkit.MimeTypeMap;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.sensor.uploader.UploadTask;

import java.io.File;

import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

public class NetworkUtils {
    private static final String TAG = "NetworkUtils";
    private static final String ROOT_URL = BuildConfig.WEB_SERVER;
    private static final String COLLECTED_DATA_URL = ROOT_URL + "/collected_data";

    private static final Gson gson = new Gson();
    private static final OkHttpClient client = new OkHttpClient();
    public static final String MIME_TYPE_JSON = "application/json";
    public static final String MIME_TYPE_BIN = "application/octet-stream";
    public static final String MIME_TYPE_MP3 = "audio/mpeg";
    public static final String MIME_TYPE_TXT = "text/plain";
    public static final String MIME_TYPE_ZIP = "application/zip";

    /*
        fileType:
            - 0 sensor bin
     */
    /*
    public static void uploadCollectedData(File file, int fileType, String name, String userId, long timestamp, String commit, Callback callback) {
        String extension = MimeTypeMap.getFileExtensionFromUrl(Uri.fromFile(file).toString());
        String mime = null;
        if(MimeTypeMap.getSingleton().hasMimeType(extension)==false){
            switch (extension.toLowerCase()) {
                case "json":
                    mime = MIME_TYPE_JSON;
                    break;
                case "bin":
                    mime = MIME_TYPE_BIN;
                    break;
                case "mp3":
                    mime = MIME_TYPE_MP3;
                    break;
                case "txt":
                    mime = MIME_TYPE_TXT;
                    break;
            }
        }
        else
            mime = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension.toLowerCase());
        Log.e("upload:","extension:"+extension+" mime:"+mime);
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", file.getName(), RequestBody.create(MediaType.parse(mime), file))
                .addFormDataPart("fileType", String.valueOf(fileType))
                .addFormDataPart("userId", userId)
                .addFormDataPart("name", name)
                .addFormDataPart("timestamp", String.valueOf(timestamp))
                .addFormDataPart("commit", commit)
                .build();
        Request request = new Request.Builder()
                .url(COLLECTED_DATA_URL)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .post(requestBody)
                .build();
        client.newCall(request).enqueue(callback);
    }
     */

    public static void uploadCollectedData(UploadTask task, Callback callback) {
        String extension = MimeTypeMap.getFileExtensionFromUrl(Uri.fromFile(task.getFile()).toString());
//        String mime = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension.toLowerCase());
        String mime;
        if (!MimeTypeMap.getSingleton().hasMimeType(extension)) {
            switch (extension.toLowerCase()) {
                case "json":
                case "meta":
                    mime = MIME_TYPE_JSON;
                    break;
                case "mp3":
                    mime = MIME_TYPE_MP3;
                    break;
                case "txt":
                    mime = MIME_TYPE_TXT;
                    break;
                case "zip":
                    mime = MIME_TYPE_ZIP;
                    break;
                case "bin":
                default:
                    mime = MIME_TYPE_BIN;
            }
        } else {
            mime = MimeTypeMap.getSingleton().getMimeTypeFromExtension(extension.toLowerCase());
        }
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", task.getFile().getName(), RequestBody.create(MediaType.parse(mime), task.getFile()))
                .addFormDataPart("meta", gson.toJson(task.getMeta()))
                .build();
        Request request = new Request.Builder()
                .url(COLLECTED_DATA_URL)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json")
                .post(requestBody)
                .build();
        client.newCall(request).enqueue(callback);
    }
}
