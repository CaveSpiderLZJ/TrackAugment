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
import com.hcifuture.datacollection.ui.ConfigTaskActivity;
import com.hcifuture.datacollection.ui.NormalAlertDialog;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class TaskListAdapter extends BaseAdapter {
    private Context mContext;
    private RootListBean mRootList;
    private LayoutInflater mInflater;

    public TaskListAdapter(Context context, RootListBean rootList) {
        mContext = context;
        mRootList = rootList;
        mInflater = LayoutInflater.from(context);
    }

    @Override
    public int getCount() {
        return mRootList.getTaskLists().size();
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
        view = mInflater.inflate(R.layout.fragment_task_list, null);
        TextView taskListId = view.findViewById(R.id.task_list_id);
        TextView taskListName = view.findViewById(R.id.task_list_name);
        Button btnDelete = view.findViewById(R.id.btn_delete);

        RootListBean.TaskList taskList = mRootList.getTaskLists().get(i);
        taskListId.setText("任务列表ID：" + taskList.getId());
        taskListName.setText("任务列表名称：" + taskList.getName());

        btnDelete.setOnClickListener((v) -> {
            NormalAlertDialog dialog = new NormalAlertDialog(mContext,
                    "Delete task list: " + taskList.getId() + "?", "");
            dialog.setPositiveButton("Yes",
                    (dialogInterface, i1) -> {
                        String id = taskList.getId();
                        for (int j = 0; j < mRootList.getTaskLists().size(); j++) {
                            if (mRootList.getTaskLists().get(j).getId().equals(id)) {
                                mRootList.deleteTaskList(j);
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

        // use bundle to add extra parameters
        view.setOnClickListener((v) -> {
            Bundle bundle = new Bundle();
            bundle.putInt("task_list_idx", i);
            Intent intent = new Intent(mContext, ConfigTaskActivity.class);
            intent.putExtras(bundle);
            mContext.startActivity(intent);
        });
        return view;
    }
}
