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
public class AddTaskListActivity extends AppCompatActivity {

    private AppCompatActivity mActivity;
    private Context mContext;
    private EditText mEditTextName;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_task_list);

        mActivity = this;
        mContext = this;

        mEditTextName = findViewById(R.id.add_task_list_edit_text_name);
        Button btnAdd = findViewById(R.id.add_task_list_btn_add);
        Button btnCancel = findViewById(R.id.add_task_list_btn_cancel);
        btnAdd.setOnClickListener((v) -> addTaskList());
        btnCancel.setOnClickListener((v) -> this.finish());
    }

    private void addTaskList() {
        Log.d("AddTaskListActivity", "addNewTask() called");
        NetworkUtils.getRootList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                RootListBean rootList = new Gson().fromJson(response.body(), RootListBean.class);
                RootListBean.TaskList taskList = new RootListBean.TaskList(
                        RandomUtils.generateRandomTaskListId(), mEditTextName.getText().toString());
                rootList.addTaskList(taskList);
                NetworkUtils.updateRootList(mContext, rootList, 0, new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) { mActivity.finish(); }
                });
            }
        });
    }
}