document.addEventListener('DOMContentLoaded', function () {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    // Show first tab by default
    tabContents.forEach((content, i) => {
        content.style.display = (i === 0) ? '' : 'none';
    });

    tabs.forEach((tab, idx) => {
        tab.addEventListener('click', function() {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        tabContents.forEach((content, i) => {
            content.style.display = (tab.getAttribute('data-tab') === content.id) ? '' : 'none';
        });
        });
    });
});