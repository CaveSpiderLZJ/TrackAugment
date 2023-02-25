package com.hcifuture.datacollection.ui.adapter;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.ui.ModifySubtaskActivity;
import com.hcifuture.datacollection.ui.NormalAlertDialog;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.util.List;

public class SubtaskAdapter extends BaseAdapter {
    private Context mContext;
    private RootListBean mRootList;
    private LayoutInflater mInflater;
    private int mTaskListIdx;
    private int mTaskIdx;

    public SubtaskAdapter(Context context, RootListBean rootList, int taskListIdx, int taskIdx) {
        mContext = context;
        mRootList = rootList;
        mInflater = LayoutInflater.from(context);
        mTaskListIdx = taskListIdx;
        mTaskIdx = taskIdx;
    }

    @Override
    public int getCount() {
        return mRootList.getTaskLists().get(mTaskListIdx)
                .getTasks().get(mTaskIdx).getSubtasks().size();
    }

    @Override
    public Object getItem(int i) {
        return null;
    }

    @Override
    public long getItemId(int i) {
        return 0;
    }

    @Override
    public View getView(int i, View view, ViewGroup viewGroup) {
        view = mInflater.inflate(R.layout.fragment_subtask, null);
        TextView subtaskId = view.findViewById(R.id.subtask_id);
        TextView subtaskName = view.findViewById(R.id.subtask_name);
        TextView subtaskTimes = view.findViewById(R.id.subtask_times);
        TextView subtaskDuration = view.findViewById(R.id.subtask_duration);
        Button btnDelete = view.findViewById(R.id.btn_delete);

        RootListBean.TaskList.Task.Subtask subtask = mRootList
                .getTaskLists().get(mTaskListIdx).getTasks().get(mTaskIdx).getSubtasks().get(i);
        subtaskId.setText("子任务ID：" + subtask.getId());
        subtaskName.setText("子任务名称：" + subtask.getName());
        subtaskTimes.setText("录制次数：" + subtask.getTimes());
        subtaskDuration.setText("单次时长：" + subtask.getDuration() + " ms");

        btnDelete.setOnClickListener((v) -> {
            NormalAlertDialog dialog = new NormalAlertDialog(mContext,
                    "Delete subtask: " + subtask.getId() + "?", "");
            dialog.setPositiveButton("Yes",
                    (dialogInterface, i1) -> {
                        String id = subtask.getId();
                        List<RootListBean.TaskList.Task.Subtask> subtasks = mRootList
                                .getTaskLists().get(mTaskListIdx).getTasks().get(mTaskIdx).getSubtasks();
                        for (int j = 0; j < subtasks.size(); j++) {
                            if (subtasks.get(j).getId().equals(id)) {
                                mRootList.getTaskLists().get(mTaskListIdx).getTasks()
                                        .get(mTaskIdx).deleteSubtask(j);
                                break;
                            }
                        }
                        NetworkUtils.updateRootList(mContext, mRootList, 0, new StringCallback() {
                            @Override
                            public void onSuccess(Response<String> response) {}
                        });
                        this.notifyDataSetChanged();
                    });
            dialog.setNegativeButton("No", (dialogInterface, i12) -> dialog.dismiss());
            dialog.create();
            dialog.show();
        });

        view.setOnClickListener((v) -> {
            Bundle bundle = new Bundle();
            bundle.putInt("task_list_idx", mTaskListIdx);
            bundle.putInt("task_idx", mTaskIdx);
            bundle.putInt("subtask_idx", i);
            Intent intent = new Intent(mContext, ModifySubtaskActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });

        return view;
    }
}
