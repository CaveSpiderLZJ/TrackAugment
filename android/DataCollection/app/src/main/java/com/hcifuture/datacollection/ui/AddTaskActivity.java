package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.RandomUtils;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.StringListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity used for adding a new task by users.
 * Jumped from ConfigTaskActivity.
 */
public class AddTaskActivity extends AppCompatActivity {

    private AppCompatActivity mActivity;
    private Context mContext;
    private EditText mEditTextName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_task);

        mActivity = this;
        mContext = this;

        Bundle bundle = getIntent().getExtras();
        int taskListIdx = bundle.getInt("task_list_idx");

        mEditTextName = findViewById(R.id.add_task_edit_text_name);
        Button btnAdd = findViewById(R.id.add_task_btn_add);
        Button btnCancel = findViewById(R.id.add_task_btn_cancel);
        btnAdd.setOnClickListener((v) -> addTask(taskListIdx));
        btnCancel.setOnClickListener((v) -> this.finish());
    }

    private void addTask(int taskListIdx) {
        Log.d("AddTaskActivity", "addTask() called");
        NetworkUtils.getRootList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                RootListBean rootList = new Gson().fromJson(response.body(), RootListBean.class);
                RootListBean.TaskList.Task task = new RootListBean.TaskList.Task(
                        RandomUtils.generateRandomTaskId(), mEditTextName.getText().toString());
                rootList.getTaskLists().get(taskListIdx).addTask(task);
                NetworkUtils.updateRootList(mContext, rootList, 0, new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) { mActivity.finish(); }
                });
            }
        });
    }
}