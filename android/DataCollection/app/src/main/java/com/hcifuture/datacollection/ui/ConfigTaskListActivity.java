package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.ui.adapter.TaskListAdapter;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity to configure task settings.
 * Jumped from MainActivity.
 */
public class ConfigTaskListActivity extends AppCompatActivity {

    private Context mContext;
    private ListView mTaskListListView;
    private TaskListAdapter mTaskListAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_task_list);
        mContext = this;
        mTaskListListView = findViewById(R.id.config_task_list_list_view);

        Button btnAdd = findViewById(R.id.config_task_list_btn_add);
        btnAdd.setOnClickListener((v) -> {
            Intent intent = new Intent(ConfigTaskListActivity.this, AddTaskListActivity.class);
            startActivity(intent);
        });

        Button btnBack = findViewById(R.id.config_task_list_btn_back);
        btnBack.setOnClickListener((v) -> this.finish());
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
                mTaskListAdapter = new TaskListAdapter(mContext, rootList);
                mTaskListListView.setAdapter(mTaskListAdapter);
            }
        });
    }
}