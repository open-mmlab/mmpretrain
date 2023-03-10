var collapsedSections = ['Advanced Guides', 'Model zoo', 'Visualization', 'Analysis Tools', 'Deployment', 'Notes'];

$(document).ready(function () {
  $('.model-summary').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": []
  });
});
