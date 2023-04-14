var collapsedSections = ['Advanced Guides', 'Model Zoo', 'Visualization', 'Analysis Tools', 'Deployment', 'Notes'];

$(document).ready(function () {
  $('.model-summary').DataTable({
    "stateSave": false,
    "lengthChange": false,
    "pageLength": 20,
    "order": []
  });
});
