/* func to switch between win/macos/linux  */
document.querySelectorAll(".platform").forEach(btn => {
    btn.addEventListener("click", (e) => {
        document.querySelectorAll(".platform").forEach(b => {
            b.classList.toggle("active", b === btn);
            b.setAttribute("aria-selected", b === btn ? "true" : "false");
        })
        const os = btn.dataset.os;
        document.querySelectorAll(".device").forEach(d => {
            d.classList.toggle("hidden", d.dataset.os !== os);
        });
    });
});
