var collapsedSections = ['进阶教程', '模型库', '可视化', '分析工具', '部署', '其他说明'];

$(document).ready(function () {
  $('.model-summary').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": [],
    "language": {
      "info": "显示 _START_ 至 _END_ 条目（总计 _TOTAL_ ）",
      "infoFiltered": "（筛选自 _MAX_ 条目）",
      "search": "搜索：",
      "zeroRecords": "没有找到任何条目",
      "paginate": {
        "next": "下一页",
        "previous": "上一页"
      },
    }
  });
});
