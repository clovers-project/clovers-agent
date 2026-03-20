export function creatModal() {
    const backdrop = document.createElement("div");
    backdrop.className = "backdrop";
    backdrop.onclick = () => document.body.removeChild(backdrop);
    const modal = document.createElement("div");
    modal.className = "modal";
    modal.onclick = (e) => e.stopPropagation();
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);
    return { backdrop, modal }
}