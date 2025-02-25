#ifndef PLATERWORKER_HPP
#define PLATERWORKER_HPP

#include <map>

#include "Worker.hpp"
#include "BusyCursorJob.hpp"

#include "slic3r/GUI/GUI.hpp"
#include "slic3r/GUI/GUI_App.hpp"
#include "slic3r/GUI/I18N.hpp"
#include "slic3r/GUI/Plater.hpp"
#include "slic3r/GUI/GLCanvas3D.hpp"

namespace Slic3r { namespace GUI {

class Plater;

template<class WorkerSubclass>
class PlaterWorker: public Worker {
    WorkerSubclass m_w;
    Plater *m_plater;

    class PlaterJob : public Job {
        std::unique_ptr<Job> m_job;
        Plater *m_plater;

    public:
        void process(Ctl &c) override
        {
            // Ensure that wxWidgets processing wakes up to handle outgoing
            // messages in plater's wxIdle handler. Otherwise it might happen
            // that the message will only be processed when an event like mouse
            // move comes along which might be too late.
            struct WakeUpCtl: Ctl {
                Ctl &ctl;
                WakeUpCtl(Ctl &c) : ctl{c} {}

                void update_status(int st, const std::string &msg = "") override
                {
                    wxWakeUpIdle();
                    ctl.update_status(st, msg);
                }

                bool was_canceled() const override { return ctl.was_canceled(); }

                std::future<void> call_on_main_thread(std::function<void()> fn) override
                {
                    wxWakeUpIdle();
                    return ctl.call_on_main_thread(std::move(fn));
                }

            } wctl{c};

            CursorSetterRAII busycursor{wctl};
            m_job->process(wctl);
        }

        void finalize(bool canceled, std::exception_ptr &eptr) override
        {
            m_job->finalize(canceled, eptr);

            if (eptr) try {
                std::rethrow_exception(eptr);
            }  catch (std::exception &e) {
                show_error(m_plater, _L("An unexpected error occured: ") + e.what());
                eptr = nullptr;
            }
        }

        PlaterJob(Plater *p, std::unique_ptr<Job> j)
            : m_job{std::move(j)}, m_plater{p}
        {
            // TODO: decide if disabling slice button during UI job is what we
            // want.
            //        if (m_plater)
            //            m_plater->sidebar().enable_buttons(false);
        }

        ~PlaterJob() override
        {
            // TODO: decide if disabling slice button during UI job is what we want.

            // Reload scene ensures that the slice button gets properly
            // enabled or disabled after the job finishes, depending on the
            // state of slicing. This might be an overkill but works for now.
            //        if (m_plater)
            //            m_plater->canvas3D()->reload_scene(false);
        }
    };

public:

    template<class... WorkerArgs>
    PlaterWorker(Plater *plater, WorkerArgs &&...args)
        : m_w{std::forward<WorkerArgs>(args)...}, m_plater{plater}
    {
        // Ensure that messages from the worker thread to the UI thread are
        // processed continuously.
        plater->Bind(wxEVT_IDLE, [this](wxIdleEvent &) {
            process_events();
        });
    }

    // Always package the job argument into a PlaterJob
    bool push(std::unique_ptr<Job> job) override
    {
        return m_w.push(std::make_unique<PlaterJob>(m_plater, std::move(job)));
    }

    bool is_idle() const override { return m_w.is_idle(); }
    void cancel() override { m_w.cancel(); }
    void cancel_all() override { m_w.cancel_all(); }
    void process_events() override { m_w.process_events(); }
    bool wait_for_current_job(unsigned timeout_ms = 0) override
    {
        return m_w.wait_for_current_job(timeout_ms);
    }
    bool wait_for_idle(unsigned timeout_ms = 0) override
    {
        return m_w.wait_for_idle(timeout_ms);
    }
};

}} // namespace Slic3r::GUI

#endif // PLATERJOB_HPP
