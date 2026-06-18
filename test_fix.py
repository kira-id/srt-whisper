from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import jinja2

orig_load_template = jinja2.environment.Environment._load_template

def patched_load_template(self, name, globals):
    cache_key = (id(name),)
    if self.cache is not None:
        template = self.cache.get(cache_key)
    else:
        template = None
    if template is not None and (not self.auto_reload or template.is_up_to_date):
        if globals:
            template.globals.update(globals)
        return template
    template = self.loader.load(self, name, self.make_globals(globals))
    if self.cache is not None:
        self.cache[cache_key] = template
    return template

jinja2.environment.Environment._load_template = patched_load_template

app = FastAPI()
templates = Jinja2Templates(directory='templates')

@app.get('/')
async def root(request: Request):
    return templates.TemplateResponse(request, 'index.html', {'request': request})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765, log_level='error')
