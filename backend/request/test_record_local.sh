curl -X DELETE http://localhost:60010/record -F "tasklistId=TL13r912je" -F "taskId=TKr23u1o3i" -F "subtaskId=STt3i2oi5m" -F "recordId=RD42ifoti2"

curl http://localhost:60010/record -F "tasklistId=TL13r912je" -F "taskId=TKr23u1o3i" -F "subtaskId=STt3i2oi5m" -F "recordId=RD42ifoti2" -F "timestamp=1234567890"
curl http://localhost:60010/record_file -F "tasklistId=TL13r912je" -F "taskId=TKr23u1o3i" -F "subtaskId=STt3i2oi5m" -F "recordId=RD42ifoti2" -F "timestamp=1234567890" -F "file=@Sensor_tap_3_0_220213165225.json" -F "fileType=0" -F "timestamp=2222222222"
curl http://localhost:60010/record_file -F "tasklistId=TL13r912je" -F "taskId=TKr23u1o3i" -F "subtaskId=STt3i2oi5m" -F "recordId=RD42ifoti2" -F "timestamp=1234567890" -F "file=@Timestamp_tap_3_0_220213165225.json" -F "fileType=1" -F "timestamp=2222222222"


curl http://localhost:60010/record -F "tasklistId=TL13r912je" -F "taskId=TKo4k24i9c" -F "subtaskId=STgi3m142j" -F "recordId=RDgi2mcorj" -F "timestamp=2134000000"
curl http://localhost:60010/record_file -F "tasklistId=TL13r912je" -F "taskId=TKo4k24i9c" -F "subtaskId=STgi3m142j" -F "recordId=RDgi2mcorj" -F "timestamp=1234567890" -F "file=@Sensor_a_2_0_220210151300.json" -F "fileType=0" -F "timestamp=2222255555"
curl http://localhost:60010/record_file -F "tasklistId=TL13r912je" -F "taskId=TKo4k24i9c" -F "subtaskId=STgi3m142j" -F "recordId=RDgi2mcorj" -F "timestamp=1234567890" -F "file=@Timestamp_a_2_0_220210151300.json" -F "fileType=1" -F "timestamp=2222255555"

curl http://localhost:60010/train -F "tasklistId=TL13r912je" -F "taskId=TKo4k24i9c,TKr23u1o3i" -F "timestamp=9087654321"
