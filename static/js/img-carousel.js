document.querySelectorAll('.carousel-container').forEach(function(carousel) {
    const images = carousel.querySelectorAll('.carousel-image');
    let current = 0;
    function show(idx) {
        images.forEach((img, i) => img.style.display = (i === idx) ? 'block' : 'none');
    }
    carousel.querySelector('.carousel-arrow.left').onclick = function() {
        current = (current === 0 ? images.length - 1 : current - 1);
        show(current);
    };
    carousel.querySelector('.carousel-arrow.right').onclick = function() {
        current = (current === images.length - 1 ? 0 : current + 1);
        show(current);
    };
    show(0);
});