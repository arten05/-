import random
import numpy as np
import matplotlib.pyplot as plt


def play_game(cost, deposit):
    initial_deposit = deposit
    deposit_history = [initial_deposit]

    deposit -= cost
    deposit_history.append(deposit)

    toss_count = 0
    win_amount = 0

    while True:
        toss_count += 1
        if random.choice([True, False]):
            win_amount = 2 ** toss_count
            deposit += win_amount
            deposit_history.append(deposit)
            result = win_amount >= cost
            return deposit_history, toss_count, win_amount, result
        else:
            deposit_history.append(deposit)


def run_saint_petersburg_paradox():
    COST_PER_GAME = 20
    DEPOSITS = [100, 1000, 100_000, 1_000_000]

    print(f"{'=' * 80}")
    print("ЛАБОРАТОРНАЯ РАБОТА: САНКТ-ПЕТЕРБУРГСКИЙ ПАРАДОКС")
    print(f"Стоимость игры: {COST_PER_GAME} рублей")
    print(f"{'=' * 80}")

    for deposit in DEPOSITS:
        print(f"\n{'=' * 80}")
        print(f"АНАЛИЗ ДЛЯ ДЕПОЗИТА: {deposit:,} РУБ")
        print(f"{'=' * 80}")

        print("\n1. График динамики одной игры:")
        deposit_history, tosses, win_amount, result = play_game(COST_PER_GAME, deposit)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(deposit_history)), deposit_history, 'b-', linewidth=2)
        plt.xlabel('Шаг игры', fontsize=12)
        plt.ylabel('Депозит (руб)', fontsize=12)
        plt.title(f'Динамика одной игры (Депозит: {deposit:,} руб)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=deposit_history[0], color='g', linestyle='--', alpha=0.7,
                    label=f'Начало: {deposit_history[0]:,} руб')
        plt.axhline(y=deposit_history[-1], color='r' if not result else 'g', linestyle='--', alpha=0.7,
                    label=f'Конец: {deposit_history[-1]:,} руб')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\n2. Столбчатая диаграмма для 20 игр:")
        tosses_list = []
        results_list = []

        for _ in range(20):
            _, tosses, _, result = play_game(COST_PER_GAME, deposit)
            tosses_list.append(tosses)
            results_list.append(result)

        fig, ax = plt.subplots(figsize=(12, 6))
        games = list(range(1, 21))
        colors = ['green' if res else 'red' for res in results_list]
        bars = ax.bar(games, tosses_list, color=colors, edgecolor='black')

        ax.set_xlabel('Номер игры', fontsize=12)
        ax.set_ylabel('Количество шагов до результата', fontsize=12)
        ax.set_title(f'20 игр (Депозит: {deposit:,} руб)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(games)

        max_tosses = max(tosses_list) if tosses_list else 1
        ax.set_yticks(range(0, max_tosses + 2))

        for bar, toss in zip(bars, tosses_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                    str(toss), ha='center', va='bottom', fontsize=10, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Выигрыш'),
            Patch(facecolor='red', edgecolor='black', label='Проигрыш')
        ]
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        plt.show()

        print("\n3. Статистика для 1000 игр:")
        wins = losses = total_win_amount = 0

        for _ in range(1000):
            _, _, win_amount, result = play_game(COST_PER_GAME, deposit)
            total_win_amount += win_amount
            wins += 1 if result else 0
            losses += 0 if result else 1

        win_percentage = wins / 10
        loss_percentage = losses / 10
        avg_win = total_win_amount / 1000

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

        labels = [f'Выигрыши\n{wins} игр', f'Проигрыши\n{losses} игр']
        sizes = [wins, losses]
        colors = ['#4CAF50', '#F44336']

        wedges, texts, autotexts = ax1.pie(sizes, colors=colors, startangle=90,
                                           autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100 * 1000)} игр)',
                                           textprops={'fontsize': 11})

        ax1.set_title(f'Статистика 1000 игр\nДепозит: {deposit:,} руб', fontsize=14, fontweight='bold')
        ax1.axis('equal')

        ax2.axis('off')
        stats_text = (
            f"СТАТИСТИКА 1000 ИГР\n\n"
            f"Начальный депозит: {deposit:,} руб\n\n"
            f"Выигрыши: {wins} игр\n"
            f"  ({win_percentage:.1f}%)\n\n"
            f"Проигрыши: {losses} игр\n"
            f"  ({loss_percentage:.1f}%)\n\n"
            f"Средний выигрыш:\n"
            f"  {avg_win:.2f} руб\n\n"
            f"Стоимость игры:\n"
            f"  {COST_PER_GAME} руб"
        )

        ax2.text(0.1, 0.95, stats_text, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4CAF50', edgecolor='black', label='Выигрыши'),
            Patch(facecolor='#F44336', edgecolor='black', label='Проигрыши')
        ]
        ax2.legend(handles=legend_elements, loc='lower center', fontsize=11,
                   bbox_to_anchor=(0.5, 0.05))

        plt.suptitle(f'Санкт-Петербургский парадокс | Стоимость игры: {COST_PER_GAME} руб',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def kinetic_monte_carlo(A0, B0, k1, k2, max_time=30, num_simulations=5):
    print(f"Параметры моделирования:")
    print(f"Начальные значения: A0 = {A0}, B0 = {B0}")
    print(f"Константы скоростей: k1 = {k1}, k2 = {k2}")
    print(f"Количество симуляций: {num_simulations}")
    print(f"Максимальное время: {max_time}")
    print("-" * 50)

    all_results = []

    for sim in range(num_simulations):
        print(f"\nСимуляция {sim + 1}/{num_simulations}...")

        A = A0
        B = B0
        time = 0

        A_history = [A]
        B_history = [B]
        time_history = [time]

        step = 0
        while time < max_time and (A > 0 or B > 0):
            step += 1

            a1 = k1 * A
            a2 = k2 * B
            a0 = a1 + a2

            if a0 == 0:
                break

            r1 = random.random()
            r2 = random.random()

            tau = -np.log(r1) / a0
            time += tau

            if r2 * a0 < a1:
                A -= 1
            else:
                B -= 1

            A_history.append(A)
            B_history.append(B)
            time_history.append(time)

        all_results.append((time_history, A_history, B_history))
        print(f"  Количество шагов: {step}")
        print(f"  Конечное время: {time:.2f}")
        print(f"  Конечные значения: A = {A}, B = {B}")

    return all_results


def plot_individual_simulation(times, A, B, sim_num, k1, k2):
    plt.figure(figsize=(10, 6))

    plt.step(times, A, where='post', linewidth=2, color='blue', label=f'A(t)')
    plt.step(times, B, where='post', linewidth=2, color='red', label=f'B(t)')

    plt.xlabel('Время', fontsize=14, fontweight='bold')
    plt.ylabel('Количество вещества', fontsize=14, fontweight='bold')
    plt.title(f'Симуляция {sim_num}: Эволюция системы\nk1 = {k1}, k2 = {k2}',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\nСтатистика симуляции {sim_num}:")
    print(f"  Начальные значения: A={A[0]}, B={B[0]}")
    print(f"  Конечные значения: A={A[-1]}, B={B[-1]}")
    print(f"  Время моделирования: {times[-1]:.2f}")
    print(f"  Изменение A: {A[0] - A[-1]} единиц")
    print(f"  Изменение B: {B[0] - B[-1]} единиц")


def plot_comparison_simulations(all_results, k1, k2):
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    plt.subplot(2, 1, 1)
    for i, (times, A, B) in enumerate(all_results):
        plt.step(times, A, where='post', linewidth=1.5, alpha=0.7,
                 color=colors[i], label=f'Симуляция {i + 1}')

    plt.xlabel('Время', fontsize=14, fontweight='bold')
    plt.ylabel('Вещество A', fontsize=14, fontweight='bold')
    plt.title(f'Сравнение симуляций: Вещество A\nk1 = {k1}, k2 = {k2}',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.subplot(2, 1, 2)
    for i, (times, A, B) in enumerate(all_results):
        plt.step(times, B, where='post', linewidth=1.5, alpha=0.7,
                 color=colors[i], label=f'Симуляция {i + 1}')

    plt.xlabel('Время', fontsize=14, fontweight='bold')
    plt.ylabel('Вещество B', fontsize=14, fontweight='bold')
    plt.title(f'Сравнение симуляций: Вещество B\nk1 = {k1}, k2 = {k2}',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_statistical_analysis(all_results, A0, B0, k1, k2):
    plt.figure(figsize=(12, 8))

    max_time = max([max(t) for t, A, B in all_results])
    common_time = np.linspace(0, max_time, 1000)

    A_interpolated = []
    B_interpolated = []

    for times, A, B in all_results:
        A_interp = np.interp(common_time, times, A, right=A[-1])
        B_interp = np.interp(common_time, times, B, right=B[-1])
        A_interpolated.append(A_interp)
        B_interpolated.append(B_interp)

    A_mean = np.mean(A_interpolated, axis=0)
    A_std = np.std(A_interpolated, axis=0)
    B_mean = np.mean(B_interpolated, axis=0)
    B_std = np.std(B_interpolated, axis=0)

    plt.subplot(2, 1, 1)
    plt.plot(common_time, A_mean, 'b-', linewidth=3, label='A (среднее)')
    plt.fill_between(common_time, A_mean - A_std, A_mean + A_std,
                     alpha=0.3, color='blue', label='± стандартное отклонение')

    plt.plot(common_time, B_mean, 'r-', linewidth=3, label='B (среднее)')
    plt.fill_between(common_time, B_mean - B_std, B_mean + B_std,
                     alpha=0.3, color='red')

    plt.xlabel('Время', fontsize=14, fontweight='bold')
    plt.ylabel('Количество вещества', fontsize=14, fontweight='bold')
    plt.title(f'Средние значения с доверительным интервалом\nk1 = {k1}, k2 = {k2}',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.subplot(2, 1, 2)
    A_theory = A0 * np.exp(-k1 * common_time)
    B_theory = B0 * np.exp(-k2 * common_time)

    plt.plot(common_time, A_theory, 'b--', linewidth=2, label='A (теория)')
    plt.plot(common_time, A_mean, 'b-', linewidth=2, alpha=0.7, label='A (Монте-Карло)')
    plt.plot(common_time, B_theory, 'r--', linewidth=2, label='B (теория)')
    plt.plot(common_time, B_mean, 'r-', linewidth=2, alpha=0.7, label='B (Монте-Карло)')

    plt.xlabel('Время', fontsize=14, fontweight='bold')
    plt.ylabel('Количество вещества', fontsize=14, fontweight='bold')
    plt.title(f'Сравнение с теоретической моделью\nk1 = {k1}, k2 = {k2}',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ВСЕХ СИМУЛЯЦИЙ")
    print("=" * 70)

    final_A = [A[-1] for _, A, _ in all_results]
    final_B = [B[-1] for _, _, B in all_results]

    print(f"\nКонечные значения вещества A:")
    print(f"  Среднее: {np.mean(final_A):.1f} ± {np.std(final_A):.1f}")
    print(f"  Минимум: {min(final_A)}, Максимум: {max(final_A)}")

    print(f"\nКонечные значения вещества B:")
    print(f"  Среднее: {np.mean(final_B):.1f} ± {np.std(final_B):.1f}")
    print(f"  Минимум: {min(final_B)}, Максимум: {max(final_B)}")

    times_to_half_A = []
    times_to_half_B = []

    for times, A, B in all_results:
        half_A = A0 / 2
        for t, a in zip(times, A):
            if a <= half_A:
                times_to_half_A.append(t)
                break

        half_B = B0 / 2
        for t, b in zip(times, B):
            if b <= half_B:
                times_to_half_B.append(t)
                break

    if times_to_half_A:
        print(f"\nВремя до уменьшения A вдвое:")
        print(f"  Среднее: {np.mean(times_to_half_A):.2f} ± {np.std(times_to_half_A):.2f}")

    if times_to_half_B:
        print(f"\nВремя до уменьшения B вдвое:")
        print(f"  Среднее: {np.mean(times_to_half_B):.2f} ± {np.std(times_to_half_B):.2f}")

    print(f"\nТеоретические времена полураспада:")
    print(f"  Вещество A: t½ = ln(2)/k1 = {np.log(2) / k1:.2f}")
    print(f"  Вещество B: t½ = ln(2)/k2 = {np.log(2) / k2:.2f}")


def run_kinetic_monte_carlo():
    print("\n" + "=" * 80)
    print("КИНЕТИЧЕСКИЙ МЕТОД МОНТЕ-КАРЛО - ВАРИАНТ 4")
    print("=" * 80)

    A0 = 500
    B0 = 200
    k1 = 0.9
    k2 = 0.4

    print(f"\nЗАПУСК МОДЕЛИРОВАНИЯ С ПАРАМЕТРАМИ:")
    print(f"A0 = {A0}, B0 = {B0}")
    print(f"k1 = {k1}, k2 = {k2}")
    print("=" * 70)

    all_results = kinetic_monte_carlo(A0, B0, k1, k2, max_time=30, num_simulations=5)

    print("\n" + "=" * 70)
    print("ГРАФИКИ ОТДЕЛЬНЫХ СИМУЛЯЦИЙ")
    print("=" * 70)

    for i, (times, A, B) in enumerate(all_results):
        plot_individual_simulation(times, A, B, i + 1, k1, k2)

    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНЫЙ ГРАФИК ВСЕХ СИМУЛЯЦИЙ")
    print("=" * 70)
    plot_comparison_simulations(all_results, k1, k2)

    print("\n" + "=" * 70)
    print("СТАТИСТИЧЕСКИЙ АНАЛИЗ И СРАВНЕНИЕ С ТЕОРИЕЙ")
    print("=" * 70)
    plot_statistical_analysis(all_results, A0, B0, k1, k2)

    print("\n" + "=" * 70)
    print("МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


def main():
    plt.style.use('seaborn-v0_8-darkgrid')


if __name__ == "__main__":
    main()