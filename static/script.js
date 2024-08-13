document.addEventListener("DOMContentLoaded", function() {
    window.addEventListener("load", function() {
        // Artificial delay (in milliseconds)
        const delay = 8000; // 8 seconds delay
        setTimeout(function() {
            document.body.classList.add('loaded');
        }, delay);
    });
});
