package com.hcifuture.datacollection.utils;

import static android.content.Context.MODE_PRIVATE;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.data.SensorData;
import com.hcifuture.datacollection.data.SensorData1D;
import com.hcifuture.datacollection.data.SensorData3D;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Deal with file IO operations.
 */
public class FileUtils {

    public static void makeDir(String directory) {
        try {
            File file = new File(directory);
            if (!file.exists()) {
                file.mkdir();
            }
        } catch (Exception ignored) {
        }
    }

    private static File makeFile(File file) {
        try {
            if (!file.exists()) {
                file.createNewFile();
            }
        } catch (Exception ignored) {
        }
        return file;
    }

    public static void writeStringToFile(String content, File saveFile) {
        makeFile(saveFile);
        String toWrite = content + "\r\n";
        try {
            OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(saveFile));
            writer.write(toWrite);
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void writeIMUDataToFile(List<SensorData3D> accData, List<SensorData3D> magData,
            List<SensorData3D> gyroData, List<SensorData3D> linearAccData, File saveFile) {
        makeFile(saveFile);
        try {
            FileOutputStream fos = new FileOutputStream(saveFile);
            DataOutputStream dos = new DataOutputStream(fos);
            int size;
            // write accelerometer data
            size = accData.size();
            dos.writeInt(size);
            for (SensorData3D unit: accData) {
                float[] values = unit.v;
                dos.writeFloat(values[0]); dos.writeFloat(values[1]); dos.writeFloat(values[2]);
                dos.writeLong(unit.t);
            }
            // write magnetic field data
            size = magData.size();
            dos.writeInt(size);
            for (SensorData3D unit: magData) {
                float[] values = unit.v;
                dos.writeFloat(values[0]); dos.writeFloat(values[1]); dos.writeFloat(values[2]);
                dos.writeLong(unit.t);
            }
            // write gyroscope data
            size = gyroData.size();
            dos.writeInt(size);
            for (SensorData3D unit: gyroData) {
                float[] values = unit.v;
                dos.writeFloat(values[0]); dos.writeFloat(values[1]); dos.writeFloat(values[2]);
                dos.writeLong(unit.t);
            }
            // write linear accelerometer data
            size = linearAccData.size();
            dos.writeInt(size);
            for (SensorData3D unit: linearAccData) {
                float[] values = unit.v;
                dos.writeFloat(values[0]); dos.writeFloat(values[1]); dos.writeFloat(values[2]);
                dos.writeLong(unit.t);
            }
            dos.flush();
            dos.close();
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeIMUDataToFile2(List<List<SensorData>> sensorData, File saveFile) {
        makeFile(saveFile);
        try {
            FileOutputStream fos = new FileOutputStream(saveFile);
            DataOutputStream dos = new DataOutputStream(fos);
            int size;
            float[] values;
            for (List<SensorData> sensor: sensorData) {
                size = sensor.size();
                dos.writeInt(size);
                for (SensorData unit: sensor) {
                    values = unit.v;
                    for (int i = 0; i < unit.d; i++)
                        dos.writeFloat(values[i]);
                    dos.writeLong(unit.t);
                }
            }
            dos.flush(); dos.close();
            fos.flush(); fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeLightSensorDataToFile(List<SensorData1D> data, File saveFile) {
        makeFile(saveFile);
        try {
            FileOutputStream fos = new FileOutputStream(saveFile);
            DataOutputStream dos = new DataOutputStream(fos);
            int size = data.size();
            Log.d("writeLightSensorDataToFile", "data size: " + size);
            dos.writeInt(size);
            for (SensorData1D unit: data) {
                dos.writeFloat(unit.v);
                dos.writeLong(unit.t);
            }
            dos.flush();
            dos.close();
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void copy(File src, File dst) {
        try {
            InputStream in = new FileInputStream(src);
            try {
                OutputStream out = new FileOutputStream(dst);
                try {
                    byte[] buf = new byte[1024];
                    int len;
                    while ((len = in.read(buf)) > 0) {
                        out.write(buf, 0, len);
                    }
                } finally {
                    out.close();
                }
            } finally {
                in.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Parse the motion data .bin file to the following data structure:
     * {
     *     'acc_data': {'x': [...], 'y': [...], 'z': [...], 't': [...]},
     *     'mag_data': {'x': [...], 'y': [...], 'z': [...], 't': [...]},
     *     'gyro_data': {'x': [...], 'y': [...], 'z': [...], 't': [...]},
     *     'linear_acc_data': {'x': [...], 'y': [...], 'z': [...], 't': [...]},
     * }
     * @param file The The motion .bin file acquired from backend,
     *             with the following data structure:
     *             [acc_size(float), [x(float), y(float), z(float), t(long)](acc_size),
     *              mag_size(float), [x(float), y(float), z(float), t(long)](mag_size),
     *              gyro_size(float), [x(float), y(float), z(float), t(long)](gyro_size),
     *              linear_acc_size(float), [x(float), y(float), z(float), t(long)](linear_acc_size)]
     * @return
     */
    public static Map<String, Map<String, List<Double>>> loadMotionData(File file) {
        Map<String, Map<String, List<Double>>> data = new HashMap<>();
        try {
            InputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis);
            String[] dataLabels = {"acc_data", "mag_data", "gyro_data", "linear_acc_data"};
            for (String dataLabel: dataLabels) {
                int size = dis.readInt();
                List<Double> x = new ArrayList<>(); List<Double> y = new ArrayList<>();
                List<Double> z = new ArrayList<>(); List<Double> t = new ArrayList<>();
                for (int i = 0; i < size; i++) {
                    x.add((double)dis.readFloat()); y.add((double)dis.readFloat());
                    z.add((double)dis.readFloat()); t.add((double)dis.readLong());
                }
                Map<String, List<Double>> subData = new HashMap<>();
                subData.put("x", x); subData.put("y", y);
                subData.put("z", z); subData.put("t", t);
                data.put(dataLabel, subData);
            }
            dis.close();
            fis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    /**
     * Parse the light data .bin file to the following data structure:*
     * {'v': [...], 't': [...]}
     * @param file The The light .bin file acquired from backend,
     *             with the following data structure:
     *             [size(float), [v(float), t(long)](size)]
     * @return
     */
    public static Map<String, List<Double>> loadLightData(File file) {
        Map<String, List<Double>> data = new HashMap<>();
        try {
            InputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis);
            int size = dis.readInt();
            List<Double> v = new ArrayList<>(); List<Double> t = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                v.add((double)dis.readFloat()); t.add((double)dis.readLong());
            }
            data.put("v", v); data.put("t", t);
            dis.close();
            fis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public interface DownloadListener {
        void onFinished();
    }

    public interface CheckListener {
         void onChanged(List<String> changedFilename, List<String> serverMD5s);
    }

    public static void downloadFiles(Context context, List<String> filename, DownloadListener listener) {
        if (filename.isEmpty()) {
            listener.onFinished();
            return;
        }
        AtomicInteger counter = new AtomicInteger(filename.size());
        for (String name: filename) {
            NetworkUtils.downloadFile(context, name, new FileCallback() {
                @Override
                public void onSuccess(Response<File> response) {
                    File file = response.body();
                    File saveFile = new File(BuildConfig.SAVE_PATH, name);
                    FileUtils.copy(file, saveFile);
                    file.delete();
                    if (counter.decrementAndGet() == 0) {
                        listener.onFinished();
                    }
                }
            });
        }
    }

    public static void checkFiles(Context context, List<String> filename, CheckListener listener) {
        if (filename.isEmpty()) {
            listener.onChanged(new ArrayList<>(), new ArrayList<>());
            return;
        }
        StringBuilder filenameBuilder = new StringBuilder();
        for (String name: filename) {
            filenameBuilder.append(name).append(",");
        }
        NetworkUtils.getMD5(context, filenameBuilder.toString(), new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                SharedPreferences fileMD5 = context.getSharedPreferences("FILE_MD5", MODE_PRIVATE);
                String[] md5s = response.body().split(",");
                List<String> changedFilename = new ArrayList<>();
                if (md5s.length != filename.size()) {
                    Log.e("TEST", "The requested files are not available on the web server.");
                    return;
                }
                for (int i = 0; i < filename.size(); i++) {
                    String serverMD5 = md5s[i];
                    String localMD5 = fileMD5.getString(filename.get(i), null);
                    Log.e("TEST", serverMD5 + " " + localMD5);
                    if (localMD5 == null || !localMD5.equals(serverMD5)) {
                        changedFilename.add(filename.get(i));
                    }
                }
                listener.onChanged(changedFilename, Arrays.asList(md5s));
            }
        });
    }

    public static String getFileContent(String filename) {
        StringBuffer buffer = new StringBuffer();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(filename));
            String line = null;
            while ((line = reader.readLine()) != null) {
                buffer.append(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return buffer.toString();
    }

    public static List<String> readLines(String filename) {
        List<String> result = new ArrayList<>();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(filename));
            String line = null;
            while ((line = reader.readLine()) != null) {
                line = line.replaceAll("\\p{C}", "");
                if (!line.isEmpty()) {
                    result.add(line);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    public static String fileToMD5(String path) {
        try {
            InputStream inputStream = new FileInputStream(path);
            byte[] buffer = new byte[1024];
            MessageDigest digest = MessageDigest.getInstance("MD5");
            int numRead = 0;
            while (numRead != -1) {
                numRead = inputStream.read(buffer);
                if (numRead > 0)
                    digest.update(buffer, 0, numRead);
            }
            byte [] md5Bytes = digest.digest();
            return convertHashToString(md5Bytes);
        } catch (Exception e) {
            return "";
        }
    }

    private static String convertHashToString(byte[] hashBytes) {
        StringBuilder returnVal = new StringBuilder();
        for (int i = 0; i < hashBytes.length; i++) {
            returnVal.append(Integer.toString((hashBytes[i] & 0xff) + 0x100, 16).substring(1));
        }
        return returnVal.toString().toLowerCase();
    }
}
