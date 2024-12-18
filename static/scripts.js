console.log('Login status:', localStorage.getItem('loggedIn'));
document.getElementById("get-started").addEventListener("click", function () {
    console.log('Button clicked');
    const loggedIn = localStorage.getItem('loggedIn');
    const targetPage = loggedIn ? "/upload_pdf" : "/login";
    console.log(`Redirecting to: ${targetPage}`);

    document.body.classList.add('fade-out');
    setTimeout(() => {
        window.location.href = targetPage;
    }, 500);
});
localStorage.clear();
localStorage.setItem('loggedIn', 'true');


// Button interaction with haptic and sound feedback
document.querySelectorAll('button').forEach(button => {
    button.addEventListener('click', () => {
        // Haptic feedback for mobile devices
        if (window.navigator.vibrate) {
            window.navigator.vibrate(100);
        }

        // Play sound feedback (if a sound file exists)
        const audio = new Audio('click-sound.mp3');
        audio.play();

        // Button click animation
        button.classList.add('clicked');
        setTimeout(() => {
            button.classList.remove('clicked');
        }, 300);
    });
});

// Handle page load animations
window.addEventListener('load', function() {
    document.body.classList.add('loaded');
});

// Scroll-based animations
window.addEventListener('scroll', function() {
    let elements = document.querySelectorAll('.reveal');
    let windowHeight = window.innerHeight;

    elements.forEach(element => {
        let elementTop = element.getBoundingClientRect().top;
        let elementVisible = 150;

        if (elementTop < windowHeight - elementVisible) {
            element.classList.add('active');
        } else {
            element.classList.remove('active');
        }
    });
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// Smooth fade-out effect for page transitions
document.body.classList.add('fade-in');
