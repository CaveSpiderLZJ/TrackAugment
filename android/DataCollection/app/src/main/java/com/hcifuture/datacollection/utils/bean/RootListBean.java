package com.hcifuture.datacollection.utils.bean;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Stores the meta data of a task list.
 * TaskList -> List(Task) -> List(List(Subtask))
 * CAUTION: this file would be converted to json, be careful for modifying variable names!
 */


public class RootListBean implements Serializable {

    public enum FILE_TYPE {
        TIMESTAMP,  // 0
        MOTION,     // 1
        LIGHT,      // 2
        AUDIO,      // 3
        VIDEO,      // 4
    }

    private List<TaskList> tasklists;

    // getters and setters
    public List<TaskList> getTaskLists() { return tasklists; }
    public void setTaskLists(List<TaskList> _taskLists) { tasklists = _taskLists; }

    public RootListBean() { tasklists = new ArrayList<>(); }

    public String[] getTaskListNames() {
        String[] taskListNames = new String[tasklists.size()];
        for (int i = 0; i < tasklists.size(); i++) {
            TaskList t = tasklists.get(i);
            taskListNames[i] = t.getId() + "." + t.getName();
        }
        return taskListNames;
    }

    public String getTaskListNameById(String taskListId) {
        for (int i = 0; i < tasklists.size(); i++) {
            TaskList t = tasklists.get(i);
            if (t.getId().equals(taskListId)) return t.getName();
        }
        return null;
    }

    public TaskList getTaskListById(String taskListId) {
        for (int i = 0; i < tasklists.size(); i++) {
            TaskList t = tasklists.get(i);
            if (t.getId().equals(taskListId)) return t;
        }
        return null;
    }

    public void addTaskList(TaskList newTaskList) { tasklists.add(newTaskList); }

    public void deleteTaskList(int idx) { tasklists.remove(idx); }

    public static class TaskList implements Serializable {

        private String id;
        private String name;
        private List<Task> tasks;

        // getters and setters
        public String getId() { return id; }
        public String getName() { return name; }
        public List<Task> getTasks() { return tasks; }
        public void setId(String _id) { id = _id; }
        public void setName(String _name) { name = _name; }
        public void setTasks(List<Task> _tasks) { tasks = _tasks; }

        public TaskList(String _id, String _name) {
            id = _id;
            name = _name;
            tasks = new ArrayList<>();
        }

        public String[] getTaskNames() {
            String[] taskNames = new String[tasks.size()];
            for (int i = 0; i < tasks.size(); i++) {
                Task t = tasks.get(i);
                taskNames[i] = t.getId() + "." + t.getName();
            }
            return taskNames;
        }

        public String getTaskNameById(String taskId) {
            for (int i = 0; i < tasks.size(); i++) {
                Task t = tasks.get(i);
                if (t.getId().equals(taskId)) return t.getName();
            }
            return null;
        }

        public Task getTaskById(String taskId) {
            for (int i = 0; i < tasks.size(); i++) {
                Task t = tasks.get(i);
                if (t.getId().equals(taskId)) return t;
            }
            return null;
        }

        public void addTask(Task newTask) { tasks.add(newTask); }

        public void deleteTask(int idx) { tasks.remove(idx); }

        public static class Task implements Serializable {

            private String id;
            private String name;
            public List<Subtask> subtasks;

            // getters and setters
            public String getId() { return id; }
            public String getName() { return name; }
            public List<Subtask> getSubtasks() { return subtasks; }
            public void setId(String _id) { id = _id; }
            public void setName(String _name) { name = _name; }
            public void setSubtasks(List<Subtask> _subtasks) { subtasks = _subtasks; }

            public Task(String id, String name) {
                this.id = id;
                this.name = name;
                this.subtasks = new ArrayList<>();
            }

            public String[] getSubtaskNames() {
                String[] subtaskNames = new String[subtasks.size()];
                for(int i = 0; i < subtasks.size(); i++) {
                    Subtask t = getSubtasks().get(i);
                    subtaskNames[i] = t.getId() + "." + t.getName();
                }
                return subtaskNames;
            }

            public String getSubtaskNameById(String subtaskId) {
                for(int i = 0; i < subtasks.size(); i++) {
                    Subtask t = subtasks.get(i);
                    if (t.getId().equals(subtaskId)) return t.getName();
                }
                return null;
            }

            public Subtask getSubTaskById(String subtaskId) {
                for (int i = 0; i < subtasks.size(); i++) {
                    Subtask t = subtasks.get(i);
                    if (t.getId().equals(subtaskId)) return t;
                }
                return null;
            }

            public void addSubtask(Subtask newSubtask) { subtasks.add(newSubtask); }

            public void deleteSubtask(int idx) { subtasks.remove(idx); }

            public static class Subtask implements Serializable {
                private String id;
                private String name;
                private int times;
                private int duration;

                // getters and setters
                public String getId() { return id; }
                public String getName() { return name; }
                public int getTimes() { return times; }
                public int getDuration() { return duration; }
                public void setId(String _id) { id = _id; }
                public void setName(String _name) { name = _name; }
                public void setTimes(int _times) { times = _times; }
                public void setDuration(int _duration) { duration = _duration; }

                public Subtask(String _id, String _name, int _times, int _duration) {
                    id = _id;
                    name = _name;
                    times = _times;
                    duration = _duration;
                }
            }
        }
    }
}
