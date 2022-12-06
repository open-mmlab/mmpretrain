var collapsedSections = ['Useful Tools', 'Advanced Guids', 'Model zoo', 'Notes'];

$(document).ready(function () {
  $('.model-summary').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": []
  });
});
