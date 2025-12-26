import torch


def train_runner(model, trainloader, criterion, optimizer):
    model.train()
    for i, (cx, cy, tx, ty) in enumerate(trainloader):
        cx = torch.squeeze(cx, dim=0)  # (bs, n_context, x_size)
        cy = torch.squeeze(cy, dim=0)  # (bs, n_context)
        tx = torch.squeeze(tx, dim=0)  # (bs, n_target, x_size)
        ty = torch.squeeze(ty, dim=0)  # (bs, n_target)

        c_id = cy[..., 0]
        t_id = ty[..., 0]
        cy = cy[..., 1]  # (bs, n_context)
        ty = ty[..., 1]

        cy = cy.unsqueeze(dim=-1)
        mu, sigma = model(cx, cy, tx)  # (bs, n_target), (bs, n_target)

        loss = criterion(mu, sigma, ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    target_y = ty[0]
    mean_y = mu[0]
    var_y = sigma[0]
    target_id = t_id[0]
    context_id = c_id[0]

    index = target_id.argsort()
    target_id = target_id[index]
    target_y = target_y[index]
    mean_y = mean_y[index]
    var_y = var_y[index]

    train_mse = (torch.sum((target_y - mean_y) ** 2)) / len(target_y)

    return (
        mean_y,
        var_y,
        target_id,
        target_y,
        context_id,
        loss.cpu().detach().numpy(),
        train_mse,
    )


def test_runner(model, testloader, criterion):
    model.eval()
    with torch.no_grad():
        for test_cx, test_cy, test_tx, test_ty in testloader:
            test_cx = torch.squeeze(test_cx, dim=0)  # (bs, n_context, x_size)
            test_cy = torch.squeeze(test_cy, dim=0)  # (bs, n_context)
            test_tx = torch.squeeze(test_tx, dim=0)  # (bs, n_target, x_size)
            test_ty = torch.squeeze(test_ty, dim=0)  # (bs, n_target)

            test_t_id = test_ty[..., 0]
            test_cy = test_cy[..., 1]
            test_ty = test_ty[..., 1]
            test_cy = test_cy.unsqueeze(dim=-1)

            test_pred_y, test_sigma_y = model(test_cx, test_cy, test_tx)  # Test
            test_loss = criterion(test_pred_y, test_sigma_y, test_ty)
            test_target_y = test_ty[0]
            test_pred_y = test_pred_y[0]
            test_var_y = test_sigma_y[0]
            test_target_id = test_t_id[0]

    test_index = test_target_id.argsort()
    test_target_id = test_target_id[test_index]
    test_target_y = test_target_y[test_index]
    test_pred_y = test_pred_y[test_index]
    test_var_y = test_var_y[test_index]

    test_mse = (torch.sum((test_target_y - test_pred_y) ** 2)) / len(test_target_y)
    return (
        test_pred_y,
        test_var_y,
        test_target_id,
        test_target_y,
        test_loss.cpu().detach().numpy(),
        test_mse,
    )
