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
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity to add a subtask.
 * Jumped from ConfigSubtaskActivity.
 */
public class AddSubtaskActivity extends AppCompatActivity {

    private AppCompatActivity mActivity;
    private Context mContext;
    private EditText mEditTextName;
    private EditText mEditTextTimes;
    private EditText mEditTextDuration;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_subtask);

        mActivity = this;
        mContext = this;
        mEditTextName = findViewById(R.id.add_subtask_edit_text_name);
        mEditTextTimes = findViewById(R.id.add_subtask_edit_text_times);
        mEditTextDuration = findViewById(R.id.add_subtask_edit_text_duration);

        Bundle bundle = getIntent().getExtras();
        int taskListIdx = bundle.getInt("task_list_idx");
        int taskIdx = bundle.getInt("task_idx");

        Button btnAdd = findViewById(R.id.add_subtask_btn_add);
        Button btnCancel = findViewById(R.id.add_subtask_btn_cancel);
        btnAdd.setOnClickListener((v) -> {addSubtask(taskListIdx, taskIdx);});
        btnCancel.setOnClickListener((v) -> this.finish());
    }

    private void addSubtask(int taskListIdx, int taskIdx) {
        Log.d("AddSubtaskActivity", "addSubtask() called");
        NetworkUtils.getRootList(mContext, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                RootListBean rootList = new Gson().fromJson(response.body(), RootListBean.class);
                RootListBean.TaskList.Task.Subtask subtask = new RootListBean.TaskList.Task.Subtask(
                        RandomUtils.generateRandomSubtaskId(), mEditTextName.getText().toString(),
                        Integer.parseInt(mEditTextTimes.getText().toString()),
                        Integer.parseInt(mEditTextDuration.getText().toString()));
                rootList.getTaskLists().get(taskListIdx).getTasks().get(taskIdx).addSubtask(subtask);
                NetworkUtils.updateRootList(mContext, rootList, 0, new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) { mActivity.finish(); }
                });
            }
        });
    }
}