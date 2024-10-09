document.addEventListener("DOMContentLoaded", function() {
    window.addEventListener("load", function() {
        // Artificial delay (in milliseconds)
        const delay = 3000;
        setTimeout(function() {
            document.body.classList.add('loaded');
        }, delay);
    });
});
