
import { currentUser, currentGroup, groupList } from "./core"
import { chatHistoryStorage } from "./tools"
import { marked } from 'marked';

export const chatWindow = document.getElementById("chatWindow") as HTMLDivElement;
const messageInput = document.getElementById("messageInput") as HTMLTextAreaElement;
const sendBtn = document.getElementById("sendBtn") as HTMLButtonElement;
const imageUpload = document.getElementById("imageUpload") as HTMLInputElement;
const imagePreviewArea = document.getElementById("imagePreviewArea") as HTMLDivElement;
const cloversBtn = document.getElementById("cloversBtn") as HTMLButtonElement;
const clearBtn = document.getElementById("clearBtn") as HTMLButtonElement;

const renderer = new marked.Renderer();
renderer.code = ({ text, lang }: { text: string, lang?: string }) => {
    const validLang = lang || 'text';
    return `<pre class="code-block" data-lang="${validLang}">` +
        `<code>${text}</code>` +
        `<span class="code-block-copy">点击复制代码</span>` +
        `</pre>`;
};
renderer.paragraph = ({ tokens }: { tokens: any[] }) => {
    // 重新将行内 token 渲染为 HTML 字符串
    return renderer.parser.parseInline(tokens) + '<br>';
};

chatWindow.addEventListener("click", async (e) => {
    const target = e.target as Element;
    const btn = target.closest('.code-block-copy') as HTMLSpanElement;
    if (!btn) return;
    const codeElement = btn.parentElement?.querySelector('code') as HTMLElement;
    if (!codeElement) return;
    const codeText = codeElement.innerText;
    const originalText = btn.innerText;
    try {
        await navigator.clipboard.writeText(codeText);
        btn.innerText = "已复制";
    } catch (err) {
        console.error("复制失败:", err);
        btn.innerText = "复制失败";
    }
    setTimeout(() => { btn.innerText = originalText; }, 2000);
});


export interface ChatMessage {
    type: "user"; // 消息类型
    text: string; // 文本内容
    images?: string[]; // 图片URLs (base64或实际URL)
    senderId: string; // 发送者ID (系统消息可为空)
    senderName: string; // 发送者用户名
    avatar?: string;
    groupId?: string; // 群组ID (可选)
    groupAvatar?: string;
    permission?: "SuperUser" | "Owner" | "Admin" | "Member";
}
interface ConsoleMessageData {
    0: string;
    1: any;
}

export interface ConsoleMessage {
    type: "system"
    data: ConsoleMessageData;
}

let pendingImages: File[] = [];
// let messageQueue: ChatMessage[] = [];
let messageQueue: ChatMessage | null = null; //决定 messageQueue 长度为 1
let commandQueue: ChatMessage | null = null
let connect: WebSocket | null = null;

function clearInput(): void {
    messageInput.value = "";
    pendingImages = [];
    imageUpload.value = ""; // 清除文件选择器的内容
    imagePreviewArea.innerHTML = "";
    imagePreviewArea.classList.remove("active");
}
function createAvatar(url: string) {
    if (url) {
        const avatar = document.createElement("img");
        avatar.src = url;
        avatar.className = "avatar";
        return avatar;
    } else {
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        return avatar;
    }
}


async function chatMessage(msg: ChatMessage) {
    const message = document.createElement("div");
    message.className = "message";
    const messageElement = document.createElement("div");
    message.appendChild(messageElement);
    messageElement.className = "message-column";
    // 用户消息
    if (msg.senderName === currentUser.userName) {
        message.classList.add("self");
        message.appendChild(messageElement);
        message.appendChild(createAvatar(currentUser.avatar));
    } else {
        message.classList.add("other");
        message.appendChild(createAvatar(currentGroup.avatar));
        message.appendChild(messageElement);
    }
    // 用户名
    const sender = document.createElement("p");
    sender.className = "username";
    sender.textContent = msg.senderName;
    // 消息内容气泡
    const content = document.createElement("div");
    content.className = "message-content";
    // 文本内容
    if (msg.text) {
        const text = document.createElement("p");
        text.className = "message-text";
        // text.textContent = msg.text;
        text.innerHTML = await marked.parse(msg.text, { renderer });
        content.appendChild(text);
    }
    // 图片内容
    if (msg.images && msg.images.length > 0) {
        msg.images.forEach((imgUrl) => {
            const img = document.createElement("img");
            img.className = "message-image-item";
            img.src = imgUrl;
            img.alt = "聊天图片";
            img.loading = "lazy";
            // 添加点击查看大图的事件
            img.onclick = () => {
                const backdrop = document.createElement("div");
                backdrop.className = "backdrop";
                backdrop.onclick = () => document.body.removeChild(backdrop);
                backdrop.innerHTML = `<img src="${imgUrl}" style="max-width: 100%; max-height: 100%;">`;
                document.body.appendChild(backdrop);
            };
            content.appendChild(img);
        });
    }
    messageElement.appendChild(sender);
    messageElement.appendChild(content);
    chatHistoryStorage.append(currentGroup.groupId, message.outerHTML);
    chatWindow.appendChild(message);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

export function systemMessage(text: string): void {
    const message = document.createElement("div");
    message.className = "message";
    message.classList.add("system");
    message.innerHTML = `<p class="system-text">${text}</p>`;
    chatWindow.appendChild(message);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function receiveHandle(message: ChatMessage | ConsoleMessage): void {
    if (message.type === "system") {
        switch (message["data"][0]) {
            case "log": systemMessage(message["data"][1]); return;
            case 'title': {
                const title = message["data"][1];
                currentGroup.groupName = title;
                const strongElement = document.getElementById(`groupItem${currentGroup.groupId}`)?.querySelector("strong");
                if (strongElement) { strongElement.textContent = title; }
                return;
            }
        }
    }
    else chatMessage(message);
}
export function connectCloversServer(url: string): void {
    if (connect) {
        if (connect.url === url && (connect.readyState === WebSocket.OPEN || connect.readyState === WebSocket.CONNECTING)) return;
        connect.close(1000, "Changing server address");
        connect = null;
    }
    connect = new WebSocket(url);
    connect.onopen = () => {
        systemMessage(`连接成功！已连接到 ${connect!.url}`);
        if (messageQueue != null) {
            connect!.send(JSON.stringify(messageQueue));
            messageQueue = null;
        }
        if (commandQueue != null) {
            connect!.send(JSON.stringify(commandQueue));
            commandQueue = null;
        }
    };
    connect.onmessage = (event) => {
        console.log("Received message:", event.data);
        try {
            const message: ChatMessage | ConsoleMessage = JSON.parse(event.data);
            receiveHandle(message);
        } catch (e) {
            systemMessage(event.data.length > 20 ? event.data.substring(0, 18) + "..." : event.data)
            console.error("Error parsing incoming message:", e, event.data);
        }
    };
    connect.onclose = (event) => {
        let reason;
        if (event.code !== 1000 && event.code !== 1005) {
            reason = `连接异常关闭 (代码: ${event.code})`;
        } else {
            reason = event.reason || "服务器关闭";
        }
        systemMessage(`连接已断开: ${reason}`);
    };
    connect.onerror = (event) => {
        console.error("WebSocket Error:", event);
    };
}
async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text && pendingImages.length === 0) return;
    if (currentGroup.flag) {
        currentGroup.flag = false;
        sendCommand(`title ${text.length > 60 ? text.substring(0, 60) + "..." : text}`)
        const strongElement = document.getElementById(`groupItem${currentGroup.groupId}`)?.querySelector("strong");
        if (strongElement) {
            strongElement.innerHTML = '<span class="loading"></span>';
        }
    };
    const imageUrls = await Promise.all(pendingImages.map((file) => {
        return new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target?.result as string);
            reader.readAsDataURL(file);
        });
    }));
    const newMessage: ChatMessage = {
        type: "user",
        senderId: currentUser.userId,
        senderName: currentUser.userName,
        groupId: currentGroup.groupId,
        avatar: currentUser.avatar,
        groupAvatar: currentGroup.avatar,
        permission: currentUser.permission,
        text: text,
        images: imageUrls,
    };
    receiveHandle(newMessage);
    clearInput();
    messageInput.focus();
    if (connect) {
        if (connect.readyState === WebSocket.OPEN) { connect.send(JSON.stringify(newMessage)); }
        else { messageQueue = newMessage; connectCloversServer(connect.url); }
    } else systemMessage(`未连接到 Clovers Client Console`);
}

function sendCommand(command: string): void {
    const newMessage: ChatMessage = {
        type: "user",
        senderId: currentUser.userId,
        senderName: currentUser.userName,
        groupId: currentGroup.groupId,
        avatar: currentUser.avatar,
        groupAvatar: currentGroup.avatar,
        permission: currentUser.permission,
        text: `\x05\x03\x01${command}`,
        images: [],
    };
    if (connect) {
        if (connect.readyState === WebSocket.OPEN) { connect.send(JSON.stringify(newMessage)); }
        else { commandQueue = newMessage; connectCloversServer(connect.url); }
    } else systemMessage(`未连接到 Clovers Client Console`);
}


sendBtn.onclick = () => {
    messageInput.style.height = 'auto';
    sendMessage();
};
function renderImagePreview(file: File, index: number): void {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
        const item = document.createElement("div");
        item.className = "preview-item";
        const removeBtn = document.createElement("button");
        removeBtn.className = "delete-button";
        removeBtn.id = "deleteImageBtn";
        removeBtn.innerHTML = '×';
        removeBtn.title = "移除图片";
        removeBtn.onclick = () => removeImage(index);
        item.appendChild(removeBtn);
        const img = document.createElement("img");
        img.className = "preview-image";
        img.src = e.target?.result as string;
        item.appendChild(img);
        imagePreviewArea.appendChild(item);
    };
    reader.readAsDataURL(file);
}
function removeImage(index: number): void {
    pendingImages.splice(index, 1);
    // 重新渲染预览区
    imagePreviewArea.innerHTML = "";
    if (pendingImages.length > 0) { pendingImages.forEach((file, newIndex) => renderImagePreview(file, newIndex)) }
    else { imagePreviewArea.classList.remove("active"); }
}

imageUpload.onchange = (event: Event) => {
    const files = (event.target as HTMLInputElement).files;
    if (files === null) return;
    for (let i = 0; i < files.length; i++) {
        if (files[i].type.startsWith("image/")) {
            pendingImages.push(files[i]);
        }
    }
    imagePreviewArea.innerHTML = "";
    pendingImages.forEach((file, index) => renderImagePreview(file, index));
    imagePreviewArea.classList.add("active");

};

messageInput.oninput = () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
};

messageInput.onkeydown = (event: KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
        messageInput.style.height = 'auto';
        event.preventDefault();
        sendMessage();
    }
};

cloversBtn.onclick = () => {
    let url = prompt("请输入 clovers 服务器地址:", connect?.url || "ws://");
    if (url === null) {
        return;
    }
    url = url.trim();
    if (!url) {
        return;
    }
    if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
        alert("请使用 ws:// 或 wss:// 开头。");
        return;
    }
    localStorage.setItem("CloversClientConsoleUrl", url);
    connectCloversServer(url);
};

clearBtn.onclick = () => {
    chatWindow.innerHTML = "";
    chatHistoryStorage.delete(currentGroup.groupId);
    if (currentGroup.flag != true) {
        sendCommand("cleanup");
        currentGroup.flag = true;
        localStorage.setItem("groupList", JSON.stringify(groupList));
    }
};