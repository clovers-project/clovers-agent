import type { GroupInfo, } from '../types';
import { groupList, setCurrentGroup, defaultGroupInfo, currentGroup } from "../core";
import { chatWindow, systemMessage } from "../chat";
import { creatModal } from "../modal";
import { sideBarContent, sideBarTitle } from ".";
import { cropImageToSquare, chatHistoryStorage } from "../tools";



export function showGroupChatHistory(groupId: string) {
    chatHistoryStorage.get(groupId).then((history) => {
        chatWindow.innerHTML = "";
        systemMessage(`当前会话「${currentGroup.groupName}」`);
        chatWindow.innerHTML += history;
    });
}


function renderGroupItem(group: GroupInfo) {
    const groupItem = document.createElement("div");
    groupItem.className = "itemlist-item";
    groupItem.id = `groupItem${group.groupId}`;
    const avatar = group.avatar ? `<img src="${group.avatar}" class="avatar left">` : '<div class="avatar left"></div>';
    groupItem.innerHTML = `<div class="grow-flex">${avatar}<strong>${group.groupName}</strong></div>`;
    const setting = document.createElement("button");
    setting.className = "tool-btn";
    setting.innerHTML = '<i class="fa-solid fa-ellipsis-vertical"></i>';
    groupItem.appendChild(setting);
    groupItem.addEventListener("click", () => {
        if (currentGroup.groupId === group.groupId) return;
        const chatHistory = chatWindow.cloneNode(true) as HTMLDivElement;
        chatHistory.querySelectorAll('div.message.system').forEach(el => el.remove());
        showGroupChatHistory(group.groupId);
        setCurrentGroup(group.groupId);
    });
    setting.addEventListener("click", (e) => {
        e.stopPropagation();
        groupInfoTemplate(creatModal(), group);
    });
    return groupItem;
}


export function renderGroupList() {
    if (groupList.length === 0) setCurrentGroup('1');
    sideBarContent.innerHTML = "";
    groupList.forEach((group) => sideBarContent.appendChild(renderGroupItem(group)));
    sideBarTitle.innerHTML = '<div class="grow-flex"><strong>会话列表</strong></div>';;
    const addGroupBtn = document.createElement("button");
    addGroupBtn.className = "tool-btn";
    addGroupBtn.innerHTML = '<i class="fa-solid fa-plus"></i>';
    sideBarTitle.appendChild(addGroupBtn);
    addGroupBtn.onclick = () => {
        let group = { ...defaultGroupInfo, groupId: Date.now().toString() };
        groupList.push(group);
        sideBarContent.appendChild(renderGroupItem(group));
        localStorage.setItem("groupList", JSON.stringify(groupList));
    };
    // const deleteAllBtn = document.createElement("button");
    // deleteAllBtn.className = "delete-all-button";
    // deleteAllBtn.innerHTML = '<i class="fa-solid fa-trash"></i>'
    // deleteAllBtn.onclick = async () => {
    //     groupList.length = 0;
    //     setCurrentGroup("1");
    //     localStorage.setItem("groupList", JSON.stringify(groupList));
    //     sideBarContent.innerHTML = ""
    //     groupList.forEach((group) => sideBarContent.appendChild(renderGroupItem(group)));
    //     await chatHistoryStorage.clearAll();
    // };
    // const grow = document.createElement("div");
    // grow.className = "grow-flex";
    // sideBarArea.appendChild(grow);
    // sideBarArea.appendChild(deleteAllBtn);
}
function groupInfoTemplate({ backdrop, modal } = creatModal(), group: GroupInfo) {
    const content = document.createElement("div");
    content.className = "modal-content";
    content.innerHTML = `
<h3>请输入会话信息</h3>
<div class="modal-item">
    <label for="nickname">会话名称:</label>
    <input type="text" id="groupName" value="${group.groupName}">
</div>
<div class="modal-item">
    <label for="groupId">会话ID:</label>
    <input type="text" id="groupId" value="${group.groupId}">
</div>
<div class="modal-item">
    <label for="groupAvatarUrl">会话头像:</label>
    <div class="avatar-input">
        <input type="text" id="groupAvatarUrl" value="${group.avatar}" placeholder="输入图片 URL 或点击右侧按钮上传">
        <label for="groupAvatarUpload" class="cancle-button" title="上传图片">选择</label>
        <input type="file" id="groupAvatarUpload" accept="image/*" class="hidden" />
    </div>
</div>`;
    const confirmBtn = document.createElement("button");
    confirmBtn.className = "confirm-button";
    confirmBtn.textContent = "确认";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "cancle-button";
    cancelBtn.textContent = "取消";
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "delete-button";
    deleteBtn.textContent = "删除";
    const btns = document.createElement("div");
    btns.className = "modal-buttons";
    btns.appendChild(confirmBtn);
    btns.appendChild(cancelBtn);
    btns.appendChild(deleteBtn);
    content.appendChild(btns);
    modal.appendChild(content);
    confirmBtn.onclick = () => {
        group.groupName = (content.querySelector("#groupName") as HTMLInputElement).value.trim();
        group.groupId = (content.querySelector("#groupId") as HTMLInputElement).value.trim();
        group.avatar = (content.querySelector("#groupAvatarUrl") as HTMLInputElement).value.trim();
        sideBarContent.innerHTML = ""
        groupList.forEach((group) => sideBarContent.appendChild(renderGroupItem(group)));
        document.body.removeChild(backdrop);
        localStorage.setItem("groupList", JSON.stringify(groupList));
    };
    cancelBtn.onclick = () => document.body.removeChild(backdrop);
    deleteBtn.onclick = () => {
        const index = groupList.indexOf(group);
        if (index !== -1) groupList.splice(index, 1);
        chatHistoryStorage.delete(group.groupId);
        if (currentGroup.groupId === group.groupId) {
            setCurrentGroup('1');
            showGroupChatHistory(currentGroup.groupId);
        };
        sideBarContent.innerHTML = ""
        groupList.forEach((group) => sideBarContent.appendChild(renderGroupItem(group)));
        document.body.removeChild(backdrop);
        localStorage.setItem("groupList", JSON.stringify(groupList));

    };
    const groupAvatarUpload = content.querySelector("#groupAvatarUpload") as HTMLInputElement;
    groupAvatarUpload.onchange = async (event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (!file) return;
        const croppedBlob = await cropImageToSquare(file);
        const blobUrl = URL.createObjectURL(croppedBlob);
        group.avatar = blobUrl;
        (content.querySelector("#groupAvatarUrl") as HTMLInputElement).value = blobUrl;
    };
}


