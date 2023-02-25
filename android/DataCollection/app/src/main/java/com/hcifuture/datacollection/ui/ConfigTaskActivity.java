package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.ui.adapter.TaskAdapter;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity to configure task settings.
 * Jumped from MainActivity.
 */
public class ConfigTaskActivity extends AppCompatActivity {

    private Context mContext;
    private ListView mTaskListView;
    private TextView mTaskListNameView;
    private TaskAdapter mTaskAdapter;
    private int mTaskListIdx;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_task);
        mContext = this;
        mTaskListView = findViewById(R.id.config_task_list_view);
        mTaskListNameView = findViewById(R.id.config_task_title);

        Bundle bundle = getIntent().getExtras();
        mTaskListIdx = bundle.getInt("task_list_idx");

        Button btnAdd = findViewById(R.id.config_task_btn_add);
        btnAdd.setOnClickListener((v) -> {
            Bundle addBundle = new Bundle();
            addBundle.putInt("task_list_idx", mTaskListIdx);
            Intent intent = new Intent(ConfigTaskActivity.this, AddTaskActivity.class);
            intent.putExtras(addBundle);
            startActivity(intent);
        });

        Button btnBack = findViewById(R.id.config_task_btn_back);
        btnBack.setOnClickListener((v) -> this.finish());

        Button btnModify = findViewById(R.id.config_task_btn_modify);
        btnModify.setOnClickListener((v) -> {
            Bundle modifyBundle = new Bundle();
            modifyBundle.putInt("task_list_idx", mTaskListIdx);
            Intent intent = new Intent(ConfigTaskActivity.this, ModifyTaskListActivity.class);
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
                mTaskListNameView.setText("Task List: " + rootList.getTaskLists().get(mTaskListIdx).getName());
                mTaskAdapter = new TaskAdapter(mContext, rootList, mTaskListIdx);
                mTaskListView.setAdapter(mTaskAdapter);
            }
        });
    }
}