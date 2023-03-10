package com.hcifuture.datacollection.ui.adapter;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Button;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.ui.NormalAlertDialog;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.ui.ConfigSubtaskActivity;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.util.List;

public class TaskAdapter extends BaseAdapter {
    private Context mContext;
    private RootListBean mRootList;
    private LayoutInflater mInflater;
    private int mTaskListIdx;

    public TaskAdapter(Context context, RootListBean rootList, int taskListIdx) {
        mContext = context;
        mRootList = rootList;
        mInflater = LayoutInflater.from(context);
        mTaskListIdx = taskListIdx;
    }

    @Override
    public int getCount() { return mRootList.getTaskLists().get(mTaskListIdx).getTasks().size(); }

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

        Log.d("### TaskAdapter.getView", "i = " + String.valueOf(i));

        view = mInflater.inflate(R.layout.fragment_task, null);
        TextView taskId = view.findViewById(R.id.task_id);
        TextView taskName = view.findViewById(R.id.task_name);
        Button btnDelete = view.findViewById(R.id.btn_delete);

        RootListBean.TaskList.Task task = mRootList.getTaskLists().get(mTaskListIdx).getTasks().get(i);
        taskId.setText("任务ID：" + task.getId());
        taskName.setText("任务名称：" + task.getName());
        btnDelete.setOnClickListener((v) -> {
            NormalAlertDialog dialog = new NormalAlertDialog(mContext,
                    "Delete task: " + task.getId() + "?", "");
            dialog.setPositiveButton("Yes",
                    (dialogInterface, i1) -> {
                        String id = task.getId();
                        List<RootListBean.TaskList.Task> tasks = mRootList
                                .getTaskLists().get(mTaskListIdx).getTasks();
                        for (int j = 0; j < tasks.size(); j++) {
                            if (tasks.get(j).getId().equals(id)) {
                                mRootList.getTaskLists().get(mTaskListIdx).deleteTask(j);
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
            bundle.putInt("task_idx", i);
            Log.d("### TaskAdapter.getView", "when clicked, i = " + String.valueOf(i));
            Intent intent = new Intent(mContext, ConfigSubtaskActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });
        return view;
    }
}
