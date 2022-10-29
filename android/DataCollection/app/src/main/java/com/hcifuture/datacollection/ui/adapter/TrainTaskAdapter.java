package com.hcifuture.datacollection.ui.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.CheckBox;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.bean.TaskListBean;

public class TrainTaskAdapter extends BaseAdapter {
    private Context mContext;
    private TaskListBean taskList;
    private LayoutInflater inflater;
    private boolean[] selected;

    public TrainTaskAdapter(Context context, TaskListBean taskList) {
        this.mContext = context;
        this.taskList = taskList;
        this.inflater = LayoutInflater.from(context);
        this.selected = new boolean[taskList.getTasks().size()];
    }

    @Override
    public int getCount() {
        return taskList.getTasks().size();
    }

    @Override
    public Object getItem(int i) {
        return null;
    }

    @Override
    public long getItemId(int i) {
        return 0;
    }

    public boolean[] getSelected() {
        return this.selected;
    }

    @Override
    public View getView(int i, View view, ViewGroup viewGroup) {
        view = inflater.inflate(R.layout.fragment_train_task, null);
        TextView taskId = view.findViewById(R.id.taskIdTrain);
        TextView taskName = view.findViewById(R.id.taskNameTrain);
        TextView taskTimes = view.findViewById(R.id.taskTimesTrain);
        TextView taskDuration = view.findViewById(R.id.taskDurationTrain);
        TextView taskVideo = view.findViewById(R.id.taskVideoTrain);
        TextView taskAudio = view.findViewById(R.id.taskAudioTrain);

        TaskListBean.Task task = taskList.getTasks().get(i);
        taskName.setText(task.getName());
        taskId.setText("  编号:            " + task.getId());
        taskTimes.setText("  录制次数:     " + task.getTimes());
        taskDuration.setText("  单次时长:     " + task.getDuration() + " ms");
        taskVideo.setText("  开启摄像头: " + task.isVideo());
        taskAudio.setText("  开启麦克风: " + task.isAudio());

        CheckBox checkBox = view.findViewById(R.id.taskCheckBoxTrain);
        checkBox.setOnCheckedChangeListener((button, b) -> {
            this.selected[i] = b;
        });

        return view;
    }
}
