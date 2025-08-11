import torch
import torch.nn as nn

# NVTX 구간 이름을 보기 좋게 찍기 위한 helper
def nvtx_push(name): 
    if torch.cuda.is_available(): torch.cuda.nvtx.range_push(name)
def nvtx_pop():
    if torch.cuda.is_available(): torch.cuda.nvtx.range_pop()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}, cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")

    # 아주 작은 MLP
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 10),
    ).to(device)

    opt  = torch.optim.SGD(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()

    # 더미 배치 20개 정도 돌리기 (짧게)
    steps = 20
    for step in range(steps):
        # H2D: 호스트→디바이스 전송 구간 (CPU에서도 동작은 하지만 CUDA 타임라인은 비어있음)
        nvtx_push("H2D")
        x = torch.randn(256, 1024, device=device, dtype=torch.float32)
        y = torch.randint(0, 10, (256,), device=device)
        nvtx_pop()

        # FWD
        nvtx_push("FWD")
        out = model(x)
        loss = crit(out, y)
        nvtx_pop()

        # BWD+STEP
        nvtx_push("BWD+STEP")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        nvtx_pop()

        if (step + 1) % 5 == 0:
            print(f"[{step+1}] loss={loss.item():.4f}")

if __name__ == "__main__":
    main()
