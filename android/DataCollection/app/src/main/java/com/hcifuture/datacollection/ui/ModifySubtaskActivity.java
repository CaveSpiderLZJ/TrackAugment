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

/**
 * The activity to modify the setting of a subtask.
 * Jumped from the subtask in SubtaskAdapter.
 */
public class ModifySubtaskActivity extends AppCompatActivity {

    private Context mContext;
    private AppCompatActivity mActivity;
    private RootListBean mRootList;
    private EditText mEditTextName;
    private EditText mEditTextTimes;
    private EditText mEditTextDuration;
    private int mTaskListIdx;
    private int mTaskIdx;
    private int mSubtaskIdx;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modify_subtask);
        mContext = this;
        mActivity = this;
        mEditTextName = findViewById(R.id.modify_subtask_edit_text_name);
        mEditTextTimes = findViewById(R.id.modify_subtask_edit_text_times);
        mEditTextDuration = findViewById(R.id.modify_subtask_edit_text_duration);

        Bundle bundle = getIntent().getExtras();
        mTaskListIdx = bundle.getInt("task_list_idx");
        mTaskIdx = bundle.getInt("task_idx");
        mSubtaskIdx = bundle.getInt("subtask_idx");

        Button btnModify = findViewById(R.id.modify_subtask_btn_modify);
        Button btnCancel = findViewById(R.id.modify_subtask_btn_cancel);

        btnModify.setOnClickListener((v) -> modifySubtask());
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
                RootListBean.TaskList.Task.Subtask subtask = mRootList.getTaskLists()
                        .get(mTaskListIdx).getTasks().get(mTaskIdx).getSubtasks().get(mSubtaskIdx);
                mEditTextName.setText(subtask.getName());
                mEditTextTimes.setText(String.valueOf(subtask.getTimes()));
                mEditTextDuration.setText(String.valueOf(subtask.getDuration()));
            }
        });
    }

    private void modifySubtask() {
        RootListBean.TaskList.Task.Subtask subtask = mRootList.getTaskLists()
                .get(mTaskListIdx).getTasks().get(mTaskIdx).getSubtasks().get(mSubtaskIdx);
        subtask.setName(mEditTextName.getText().toString());
        subtask.setTimes(Integer.parseInt(mEditTextTimes.getText().toString()));
        subtask.setDuration(Integer.parseInt(mEditTextDuration.getText().toString()));
        NetworkUtils.updateRootList(mContext, mRootList, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) { mActivity.finish(); }
        });
    }
}