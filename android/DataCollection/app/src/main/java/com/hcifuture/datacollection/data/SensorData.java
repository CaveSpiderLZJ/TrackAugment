package com.hcifuture.datacollection.data;

/**
 * A sensor data unit with configurable dimension.
 * Stores n float sensor values and a long timestamp at the moment.
 */
public class SensorData {
    public int d;
    public float[] v;
    public long t;

    public SensorData(int dimension, float[] values, long timestamp) {
        d = dimension;
        v = values.clone();
        t = timestamp;
    }
}
