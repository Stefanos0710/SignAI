/* func to switch between win/macos/linux  */
// Nur auf der Index-Seite ausführen (nicht auf Download-Seite)
if (document.querySelector("#home") || document.querySelector("#desktop .hero-content")) {
    document.querySelectorAll(".platform").forEach(btn => {
        btn.addEventListener("click", (e) => {
            // Nur Buttons in der gleichen Section beeinflussen
            const section = btn.closest("section");

            // Toggle active state für Buttons in dieser Section
            section.querySelectorAll(".platform").forEach(b => {
                b.classList.toggle("active", b === btn);
                b.setAttribute("aria-selected", b === btn ? "true" : "false");
            });

            const os = btn.dataset.os;

            // Nur Devices in dieser Section umschalten
            section.querySelectorAll(".device").forEach(d => {
                d.classList.toggle("hidden", d.dataset.os !== os);
            });

            // Switch instruction links based on OS (nur in dieser Section)
            section.querySelectorAll(".instruction-link").forEach(link => {
                link.classList.toggle("hidden", link.dataset.osLink !== os);
            });
        });
    });
}

window.addEventListener("scroll", () => {
    const nav = document.getElementById("navbar");
    if (window.scrollY > 50) {
        nav.classList.add("scrolled");
    } else {
        nav.classList.remove("scrolled");
    };
});
