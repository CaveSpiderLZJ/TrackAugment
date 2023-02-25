package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.ui.adapter.TaskAdapter;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.ui.adapter.SubtaskAdapter;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity used to configure subtasks.
 * Jumped from the task in TaskAdapter.
 */
public class ConfigSubtaskActivity extends AppCompatActivity {

    private Context mContext;
    private ListView mSubtaskListView;
    private TextView mTaskNameView;
    private SubtaskAdapter mSubtaskAdapter;
    private int mTaskListIdx;
    private int mTaskIdx;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_subtask);
        mContext = this;
        mSubtaskListView = findViewById(R.id.config_subtask_list_view);
        mTaskNameView = findViewById(R.id.config_subtask_title);

        Bundle bundle = getIntent().getExtras();
        mTaskListIdx = bundle.getInt("task_list_idx");
        mTaskIdx = bundle.getInt("task_idx");

        Log.d("### ConfigSubtaskActivity.onCreate()", "mTaskListIdx = " + String.valueOf(mTaskListIdx)
                + ", mTaskIdx = " + String.valueOf(mTaskIdx));

        Button btnAdd = findViewById(R.id.config_subtask_btn_add);
        btnAdd.setOnClickListener((v) -> {
            Bundle addBundle = new Bundle();
            addBundle.putInt("task_list_idx", mTaskListIdx);
            addBundle.putInt("task_idx", mTaskIdx);
            Intent intent = new Intent(ConfigSubtaskActivity.this, AddSubtaskActivity.class);
            intent.putExtras(addBundle);
            startActivity(intent);
        });

        Button btnBack = findViewById(R.id.config_subtask_btn_back);
        btnBack.setOnClickListener((v) -> this.finish());

        Button btnModify = findViewById(R.id.config_subtask_btn_modify);
        btnModify.setOnClickListener((v) -> {
            Bundle modifyBundle = new Bundle();
            modifyBundle.putInt("task_list_idx", mTaskListIdx);
            modifyBundle.putInt("task_idx", mTaskIdx);
            Intent intent = new Intent(ConfigSubtaskActivity.this, ModifyTaskActivity.class);
            intent.putExtras(modifyBundle);
            startActivity(intent);
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadRootListViaNetwork();
    }

    private void loadRootListViaNetwork() {
        NetworkUtils.getRootList(this, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                RootListBean rootList = new Gson().fromJson(response.body(), RootListBean.class);
                mTaskNameView.setText("Task: " + rootList.getTaskLists().get(mTaskListIdx)
                        .getTasks().get(mTaskIdx).getName());
                mSubtaskAdapter = new SubtaskAdapter(mContext, rootList, mTaskListIdx, mTaskIdx);
                mSubtaskListView.setAdapter(mSubtaskAdapter);
            }
        });
    }
}