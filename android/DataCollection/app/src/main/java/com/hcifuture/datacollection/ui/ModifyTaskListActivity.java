package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.RootListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;


public class ModifyTaskListActivity extends AppCompatActivity {

    private Context mContext;
    private AppCompatActivity mActivity;
    private RootListBean mRootList;
    private EditText mEditTextName;
    private int mTaskListIdx;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modify_task_list);
        this.mContext = this;
        this.mActivity = this;
        mEditTextName = findViewById(R.id.modify_task_list_edit_text_name);

        Bundle bundle = getIntent().getExtras();
        mTaskListIdx = bundle.getInt("task_list_idx");

        Button btnModify = findViewById(R.id.modify_task_list_btn_modify);
        Button btnCancel = findViewById(R.id.modify_task_list_btn_cancel);

        btnModify.setOnClickListener((v) -> modifyTaskList());
        btnCancel.setOnClickListener((v) -> this.finish());
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadRootListViaNetwork();
    }

    private void loadRootListViaNetwork() {
        NetworkUtils.getRootList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mRootList = new Gson().fromJson(response.body(), RootListBean.class);
                RootListBean.TaskList taskList = mRootList.getTaskLists().get(mTaskListIdx);
                mEditTextName.setText(taskList.getName());
            }
        });
    }

    private void modifyTaskList() {
        RootListBean.TaskList taskList = mRootList.getTaskLists().get(mTaskListIdx);
        taskList.setName(mEditTextName.getText().toString());
        NetworkUtils.updateRootList(mContext, mRootList, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) { mActivity.finish(); }
        });
    }
}